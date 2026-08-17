# Qwen3.5 TP4 32K H2D Slot-Reuse Causal Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task in the
> current session. Do not use subagents.

**Goal:** Add a default-disabled, physical-slot/occupancy-generation H2D
slot-reuse diagnostic plus a diagnostic-only copy-stream wait control, with a
focused ordinary-baseline TP4/32K worker and verifier that can later determine
whether the missing read-completion-to-H2D-start edge causes the retained
batch-shape logit drift.

**Architecture:** Put immutable diagnostic state, CUDA-event ownership, slot
occupancy generations, timing resolution, and tensor-free row conversion in a
focused engine helper owned by `KVOffloadMVP0`. Wire production mapping
transitions and H2D spans through narrow manager hooks, and place one shared
read-completion marker after the final K/V slot read in decode, spec-verify,
and prefill windows. Expose explicit all-rank lifecycle methods through
`LLMEngine`, then build a separate baseline-only worker/gate/verifier that
reuses the frozen TP4/32K configuration without importing or activating paired
verify or Qwen3.5 side-state traces.

**Tech Stack:** Python 3, frozen dataclasses, PyTorch CUDA events and streams,
JSON/SHA-256, TinyLLMForge `KVOffloadMVP0`, blockwise online-softmax attention,
acknowledged TP worker commands, pytest, and the existing Qwen3.5 TP4/32K
target-KV-offload authority helpers.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify `/Users/bytedance/dev/TinyLLMForge`.
- Do not stage, commit, push, switch branches/worktrees, stash, reset, or
  clean.
- Do not use subagents.
- Do not terminate unrelated GPU processes.
- Do not run GPU, remote, NCCL, or authority workloads while implementing this
  plan; those require separate explicit authorization.
- Exact greedy parity remains mandatory.
- Keep `MAX_PROPOSAL_TOKENS=4`.
- Keep `prompt_tokens=32768`, `max_output_tokens=8`, `block_size=256`,
  `gpu_blocks=68`, `logical_blocks=640`, `tensor_parallel_size=4`,
  `kv_offload_blockwise_blocks=8`, and `enforce_eager=true`.
- Keep `kv_offload_async_copy=true`, `kv_offload_batch_copy=true`, and
  `kv_offload_writeback_on_evict=false`.
- The focused diagnostic permits only ordinary `baseline:b1` and
  `baseline:b4`; it must reject native MTP, first-target, verify-tail, paired
  traces, shadow forwards, proposal callbacks, and extra target forwards.
- Do not change verifier selection, fallback indexing, accepted-prefix
  semantics, target/proposal KV transaction semantics, recurrent side-state
  semantics, Scheduler behavior, n-gram, SAM, or unrelated MTP behavior.
- Do not change H2D/D2H pair selection, source/destination tensors,
  coalescing, production completion events, counters, bytes, batches, spans,
  eviction policy, or movement accounting.
- Mode `off` must allocate no timing events, record no rows, insert no waits,
  and add no artifact fields to ordinary authority output.
- Observe mode must never add a current-to-copy dependency.
- Control mode may add only unique predecessor
  `copy_stream.wait_event(event)` calls immediately before the affected H2D
  span starts.
- No device synchronization, current-stream synchronization, copy-stream
  synchronization, layer barrier, TP collective barrier, or process-group
  barrier may be added to the H2D hot path.
- Timing resolution may synchronize only during explicit diagnostic drain
  after generation and cleanup.
- A local implementation may establish only
  `H2D_SLOT_REUSE_DIAGNOSTIC_CONTRACT=ESTABLISHED` and
  `DEFAULT_OFF_NON_INVASIVENESS=ESTABLISHED`.
- Keep these classifications unchanged until a separately authorized real
  GPU campaign:

```text
FOCUSED_H2D_GPU_DIAGNOSTIC=NOT_APPROVED
PAIRED_TRACE_REMOTE_DIAGNOSTIC=NOT_APPROVED
TP4_32K_H2D_SLOT_REUSE_GPU_CAUSALITY=NOT_ESTABLISHED
TP4_32K_EXACT_ROOT_CAUSE=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

- Every task ends with local verification instead of a git commit.

## File Structure

- Create `tinyvllm/engine/h2d_slot_reuse_diagnostic.py`
  - Frozen row contracts, slot occupancy generations, predecessor-event
    ownership, observe/control behavior, timing resolution, and immutable
    tensor-free drain output.
- Modify `tinyvllm/engine/model_runner.py`
  - Own the recorder inside `KVOffloadMVP0`, route every slot mapping mutation
    through diagnostic lifecycle hooks, instrument unchanged H2D spans, and
    expose rank-local configure/context/drain methods.
- Modify `tinyvllm/layers/attention.py`
  - Record one read-completion event after all K/V reads for each decode,
    spec-verify, and prefill window.
- Modify `tinyvllm/engine/llm_engine.py`
  - Configure, contextualize, drain, and clear the diagnostic across all TP
    ranks using acknowledged model-runner commands.
- Create `tools/qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_gate.py`
  - Frozen constants, exact field validators, movement/config invariants,
    timing classification, and mutually exclusive terminal decisions.
- Create `tools/qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_worker.py`
  - Fresh-process baseline-only cell execution, compact prediction-index 0/1
    logits, runtime metadata, all-rank slot rows, artifact assembly, and
    failure-safe lifecycle cleanup.
- Create `tools/verify_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py`
  - Independent artifact/source/checkpoint verifier and terminal
    classification printer.
- Create `tools/test_h2d_slot_reuse_diagnostic.py`
  - CPU/dummy-event unit tests for occupancy, stream deduplication, timing,
    immutable drain, and cleanup.
- Modify `tools/test_kv_offload.py`
  - Manager integration tests proving mapping coverage, H2D span ordering,
    unchanged production events, counters, pairs, and spans.
- Modify `tools/test_blockwise_attention_planning.py`
  - Decode and prefill read-marker placement tests.
- Modify `tools/test_native_verifier_attention.py`
  - Spec-verify read-marker placement test.
- Create
  `tools/test_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py`
  - Worker, gate, all-rank lifecycle, metadata, invariant, and decision-matrix
    tests.
- Modify `AGENT_HANDOFF_STATE.md`
  - Record implementation/test evidence and preserve the no-GPU/no-causality
    claim boundary.

---

### Task 1: Add the Immutable Slot-Reuse Diagnostic Core

**Files:**
- Create: `tinyvllm/engine/h2d_slot_reuse_diagnostic.py`
- Create: `tools/test_h2d_slot_reuse_diagnostic.py`

**Interfaces:**
- Produces:
  - `H2D_SLOT_REUSE_SCHEMA: str`
  - `VALID_H2D_SLOT_REUSE_MODES: tuple[str, ...]`
  - `SlotOccupancy`
  - `SlotReadAssociation`
  - `H2DSlotOverwrite`
  - `H2DSlotReuseDrain`
  - `H2DSlotReuseDiagnostic`
  - `classify_read_h2d_ordering(delta_ms, epsilon_ms)`
- Consumed later by `KVOffloadMVP0`, `LLMEngine`, the focused worker, and the
  independent verifier.

- [x] **Step 1: Write RED tests for modes, occupancy generations, and stale-event isolation**

Create `tools/test_h2d_slot_reuse_diagnostic.py` with deterministic dummy
events and streams:

```python
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tinyvllm.engine.h2d_slot_reuse_diagnostic import (
    H2D_SLOT_REUSE_SCHEMA,
    H2DSlotReuseDiagnostic,
    classify_read_h2d_ordering,
)


class _DummyEvent:
    def __init__(self, ordinal, elapsed=None):
        self.ordinal = ordinal
        self.elapsed = {} if elapsed is None else dict(elapsed)
        self.recorded_streams = []

    def record(self, stream=None):
        self.recorded_streams.append(stream)

    def elapsed_time(self, other):
        return float(self.elapsed[other.ordinal])


class _DummyStream:
    def __init__(self, stream_id):
        self.stream_id = int(stream_id)
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event.ordinal)


class _EventFactory:
    def __init__(self):
        self.next_ordinal = 1
        self.events = {}

    def __call__(self):
        event = _DummyEvent(self.next_ordinal)
        self.events[event.ordinal] = event
        self.next_ordinal += 1
        return event


def _diagnostic(slot_count=2):
    events = _EventFactory()
    diagnostic = H2DSlotReuseDiagnostic(
        rank=0,
        slot_count=slot_count,
        event_factory=events,
        stream_id=lambda stream: stream.stream_id,
    )
    return diagnostic, events


def test_schema_and_modes_are_frozen():
    assert H2D_SLOT_REUSE_SCHEMA == (
        "qwen35.tp4-32k-h2d-slot-reuse-causal-diagnostic.v1"
    )
    diagnostic, _ = _diagnostic()
    assert diagnostic.configure("off") == {
        "rank": 0,
        "mode": "off",
    }
    with pytest.raises(ValueError, match="mode must be one of"):
        diagnostic.configure("serialize")


def test_slot_generation_increments_for_every_assignment():
    diagnostic, _ = _diagnostic(slot_count=1)
    diagnostic.configure("observe")
    first = diagnostic.assign_slot(
        physical_slot=0,
        logical_block=10,
        bound_generation=2,
    )
    second = diagnostic.assign_slot(
        physical_slot=0,
        logical_block=11,
        bound_generation=7,
    )
    third = diagnostic.assign_slot(
        physical_slot=0,
        logical_block=10,
        bound_generation=3,
    )
    assert first.occupancy_generation == 1
    assert second.occupancy_generation == 2
    assert third.occupancy_generation == 3


def test_replacement_occupancy_does_not_inherit_stale_read_event():
    diagnostic, _ = _diagnostic(slot_count=1)
    current = _DummyStream(101)
    diagnostic.configure("observe")
    first = diagnostic.assign_slot(
        physical_slot=0,
        logical_block=4,
        bound_generation=1,
    )
    diagnostic.record_read_window(
        engine_step=3,
        attention_stage="decode",
        layer_index=0,
        window_ordinal=2,
        logical_blocks=(4,),
        physical_slots=(0,),
        current_stream=current,
    )
    second = diagnostic.assign_slot(
        physical_slot=0,
        logical_block=5,
        bound_generation=1,
    )
    assert second.occupancy_generation == 2
    assert diagnostic.predecessor_event_ordinals(first) == (1,)
    assert diagnostic.predecessor_event_ordinals(second) == ()
```

- [x] **Step 2: Run the core tests RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_h2d_slot_reuse_diagnostic.py::test_schema_and_modes_are_frozen \
  tools/test_h2d_slot_reuse_diagnostic.py::test_slot_generation_increments_for_every_assignment \
  tools/test_h2d_slot_reuse_diagnostic.py::test_replacement_occupancy_does_not_inherit_stale_read_event
```

Expected: collection fails because
`tinyvllm.engine.h2d_slot_reuse_diagnostic` does not exist.

- [x] **Step 3: Implement frozen contracts and mode lifecycle**

Create `tinyvllm/engine/h2d_slot_reuse_diagnostic.py` with these public
contracts:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
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
```

Implement `H2DSlotReuseDiagnostic` with:

```python
class H2DSlotReuseDiagnostic:
    def __init__(
        self,
        *,
        rank: int,
        slot_count: int,
        event_factory: Callable[[], object],
        stream_id: Callable[[object], int],
    ):
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        self.rank = int(rank)
        self.slot_count = int(slot_count)
        self._event_factory = event_factory
        self._stream_id = stream_id
        self._mode = "off"
        self._reset_state()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def enabled(self) -> bool:
        return self._mode != "off"

    def _reset_state(self) -> None:
        self._slot_generations = [0] * self.slot_count
        self._active_occupancies = [None] * self.slot_count
        self._read_events_by_occupancy = {}
        self._read_rows = []
        self._pending_overwrites = []
        self._event_ordinals = {}
        self._next_event_ordinal = 1
        self._stream_inventory = set()
        self._context = None

    def configure(self, mode: str) -> dict:
        if mode not in VALID_H2D_SLOT_REUSE_MODES:
            raise ValueError(
                "mode must be one of off, observe, control"
            )
        if mode == "off":
            self._mode = "off"
            self._reset_state()
            return {"rank": self.rank, "mode": "off"}
        if self.enabled and (
            mode != self._mode
            or self._read_rows
            or self._pending_overwrites
        ):
            raise RuntimeError(
                "diagnostic has undrained enabled state"
            )
        if not self.enabled:
            self._reset_state()
        self._mode = mode
        return {"rank": self.rank, "mode": mode}
```

Add private validation helpers for nonnegative integer rank/slot/logical IDs,
strictly positive occupancy generations, valid stages, exact active occupancy,
unique event ordinals, and stale-generation rejection. `assign_slot()` must
increment the physical slot generation on every assignment, preserve a
snapshot of the replaced occupancy and its predecessor event map for a future
H2D span, and never attach those events to the new occupancy.

- [x] **Step 4: Add RED tests for stream supersession, distinct streams, and observe/control waits**

Append:

```python
def test_latest_read_event_supersedes_same_stream():
    diagnostic, _ = _diagnostic(slot_count=1)
    stream = _DummyStream(7)
    diagnostic.configure("observe")
    occupancy = diagnostic.assign_slot(
        physical_slot=0,
        logical_block=1,
        bound_generation=0,
    )
    for window in (0, 1):
        diagnostic.record_read_window(
            engine_step=0,
            attention_stage="decode",
            layer_index=0,
            window_ordinal=window,
            logical_blocks=(1,),
            physical_slots=(0,),
            current_stream=stream,
        )
    assert diagnostic.predecessor_event_ordinals(
        occupancy
    ) == (2,)


def test_distinct_read_streams_are_all_retained():
    diagnostic, _ = _diagnostic(slot_count=1)
    diagnostic.configure("observe")
    occupancy = diagnostic.assign_slot(
        physical_slot=0,
        logical_block=1,
        bound_generation=0,
    )
    for stream_id in (7, 9):
        diagnostic.record_read_window(
            engine_step=0,
            attention_stage="decode",
            layer_index=0,
            window_ordinal=stream_id,
            logical_blocks=(1,),
            physical_slots=(0,),
            current_stream=_DummyStream(stream_id),
        )
    assert diagnostic.predecessor_event_ordinals(
        occupancy
    ) == (1, 2)


@pytest.mark.parametrize(
    ("mode", "expected_waits"),
    (("observe", []), ("control", [1])),
)
def test_h2d_span_waits_only_in_control(mode, expected_waits):
    diagnostic, _ = _diagnostic(slot_count=1)
    current = _DummyStream(7)
    copy = _DummyStream(11)
    diagnostic.configure(mode)
    diagnostic.set_context(
        engine_step=3,
        attention_stage="decode",
        layer_index=0,
        window_ordinal=4,
    )
    diagnostic.assign_slot(
        physical_slot=0,
        logical_block=1,
        bound_generation=0,
    )
    diagnostic.record_read_window(
        engine_step=3,
        attention_stage="decode",
        layer_index=0,
        window_ordinal=4,
        logical_blocks=(1,),
        physical_slots=(0,),
        current_stream=current,
    )
    diagnostic.assign_slot(
        physical_slot=0,
        logical_block=2,
        bound_generation=0,
    )
    handle = diagnostic.begin_h2d_span(
        copy_batch_ordinal=1,
        copy_span_ordinal=0,
        pairs=((2, 0),),
        copy_stream=copy,
    )
    diagnostic.end_h2d_span(handle, copy_stream=copy)
    assert copy.waited == expected_waits
```

- [x] **Step 5: Implement read recording and H2D span ownership**

Implement:

```python
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
) -> None:
    if not self.enabled:
        return
    identities = self._validated_active_identities(
        logical_blocks,
        physical_slots,
    )
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
        self._read_rows.append(SlotReadAssociation(
            rank=self.rank,
            engine_step=int(engine_step),
            attention_stage=attention_stage,
            layer_index=int(layer_index),
            window_ordinal=int(window_ordinal),
            current_stream_id=stream,
            physical_slot=occupancy.physical_slot,
            occupancy_generation=(
                occupancy.occupancy_generation
            ),
            logical_block=occupancy.logical_block,
            bound_generation=occupancy.bound_generation,
            read_event_ordinal=ordinal,
        ))
```

`begin_h2d_span()` must:

1. Resolve each `(new_logical_block, physical_slot)` pair to the exact pending
   replacement transition for the new occupancy.
2. Collect predecessor events from the old occupancy only.
3. Deduplicate by diagnostic event ordinal across every slot in the span.
4. Call `copy_stream.wait_event(event)` only in `control`.
5. Record one timing-enabled start event on `copy_stream`.
6. Save immutable per-slot pending rows that reference the shared start event.

`end_h2d_span()` must record one timing-enabled done event on `copy_stream`,
attach it to each pending slot row, and retain event objects only until drain.
It must not read or write `KVOffloadMVP0.h2d_done` or `d2h_done`.

- [x] **Step 6: Add RED tests for timing classification, explicit no-prior rows, immutable drain, and cleanup**

Append tests that set dummy elapsed values and assert:

```python
def test_timing_classification_uses_fixed_epsilon():
    assert classify_read_h2d_ordering(0.21, 0.20) == (
        "UNSAFE_OVERLAP_OBSERVED"
    )
    assert classify_read_h2d_ordering(0.20, 0.20) == (
        "ORDERING_AMBIGUOUS"
    )
    assert classify_read_h2d_ordering(-0.21, 0.20) == (
        "READ_COMPLETED_BEFORE_H2D"
    )


def test_drain_returns_frozen_tensor_free_rows_and_releases_events():
    diagnostic, events = _diagnostic(slot_count=1)
    current = _DummyStream(3)
    copy = _DummyStream(5)
    diagnostic.configure("observe")
    diagnostic.set_context(
        engine_step=0,
        attention_stage="decode",
        layer_index=0,
        window_ordinal=0,
    )
    diagnostic.assign_slot(
        physical_slot=0,
        logical_block=1,
        bound_generation=0,
    )
    diagnostic.record_read_window(
        engine_step=0,
        attention_stage="decode",
        layer_index=0,
        window_ordinal=0,
        logical_blocks=(1,),
        physical_slots=(0,),
        current_stream=current,
    )
    diagnostic.assign_slot(
        physical_slot=0,
        logical_block=2,
        bound_generation=0,
    )
    handle = diagnostic.begin_h2d_span(
        copy_batch_ordinal=0,
        copy_span_ordinal=0,
        pairs=((2, 0),),
        copy_stream=copy,
    )
    diagnostic.end_h2d_span(handle, copy_stream=copy)
    events.events[2].elapsed[1] = 0.5
    drained = diagnostic.drain(
        synchronize=lambda: None,
        timing_epsilon_ms=0.2,
    )
    assert drained.overwrite_rows[0].timing_status == (
        "UNSAFE_OVERLAP_OBSERVED"
    )
    with pytest.raises(FrozenInstanceError):
        drained.overwrite_rows[0].control_wait_count = 9
    assert diagnostic.retained_event_count == 0


def test_disable_clears_undrained_state():
    diagnostic, _ = _diagnostic(slot_count=1)
    diagnostic.configure("observe")
    diagnostic.assign_slot(
        physical_slot=0,
        logical_block=1,
        bound_generation=0,
    )
    diagnostic.configure("off")
    assert diagnostic.retained_event_count == 0
    assert diagnostic.active_occupancies == ()
```

Add explicit tests for:

- empty slot replacement produces `NO_PRIOR_OCCUPANCY`;
- occupied slot without a read produces `NO_PRIOR_READ`;
- missing or unqueryable event timing raises a hard error;
- duplicate event ordinal raises a hard error;
- stale occupancy or generation mismatch raises a hard error;
- enabled buffer capacity overflow raises instead of dropping rows;
- drain clears rows/events but preserves the configured mode until explicit
  `configure("off")`.

- [x] **Step 7: Implement timing resolution and immutable drain**

`drain()` must:

```python
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
    return result
```

For every predecessor, compute exactly:

```python
delta_ms = h2d_start_event.elapsed_time(
    prior_read_done_event
)
```

If several predecessor events protect one destination slot, retain one
overwrite row per destination slot and classify it by the maximum positive
delta. Preserve all `read_event_ordinals` and
`control_wait_event_ordinals`; do not silently collapse timing coverage.

- [x] **Step 8: Run the complete core suite GREEN**

Run:

```bash
python3 -m pytest -q tools/test_h2d_slot_reuse_diagnostic.py
```

Expected: all tests pass without importing CUDA or allocating tensors.

---

### Task 2: Integrate Occupancy Lifecycle into `KVOffloadMVP0`

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_kv_offload.py`

**Interfaces:**
- Consumes:
  - `H2DSlotReuseDiagnostic`
  - `SlotOccupancy`
- Produces on `KVOffloadMVP0`:
  - `configure_h2d_slot_reuse_diagnostic(mode) -> dict`
  - `set_h2d_slot_reuse_context(...) -> dict`
  - `record_h2d_slot_read_window(...) -> None`
  - `drain_h2d_slot_reuse_diagnostic(...) -> dict`
  - `h2d_slot_reuse_diagnostic_summary() -> dict`
- Later consumed by blockwise attention and `LLMEngine`.

- [x] **Step 1: Write RED tests for default-off construction and explicit lifecycle**

Extend `tools/test_kv_offload.py` with a CPU-only manager built through
`KVOffloadMVP0.__new__`:

```python
def test_h2d_slot_reuse_diagnostic_defaults_off_without_events():
    manager = KVOffloadMVP0.__new__(KVOffloadMVP0)
    manager.rank = 0
    manager.gpu_blocks = 2
    manager._initialize_h2d_slot_reuse_diagnostic(
        event_factory=lambda: (_ for _ in ()).throw(
            AssertionError("event allocated in off mode")
        ),
        stream_id=id,
    )
    assert manager.h2d_slot_reuse_diagnostic_summary() == {
        "rank": 0,
        "mode": "off",
        "retained_event_count": 0,
        "read_row_count": 0,
        "overwrite_row_count": 0,
    }


def test_h2d_slot_reuse_diagnostic_lifecycle_is_explicit():
    manager = _diagnostic_manager(gpu_blocks=2)
    assert manager.configure_h2d_slot_reuse_diagnostic(
        "observe"
    ) == {"rank": 0, "mode": "observe"}
    manager.configure_h2d_slot_reuse_diagnostic("off")
    assert manager.h2d_slot_reuse_diagnostic_summary()[
        "mode"
    ] == "off"
```

Expected RED: manager lifecycle methods are absent.

- [x] **Step 2: Own the recorder and expose rank-local lifecycle methods**

Import the helper and initialize it at the end of `KVOffloadMVP0.__init__`:

```python
self._initialize_h2d_slot_reuse_diagnostic(
    event_factory=lambda: torch.cuda.Event(
        enable_timing=True
    ),
    stream_id=lambda stream: int(stream.cuda_stream),
)
```

Do not allocate an event in `_initialize_h2d_slot_reuse_diagnostic`; only
store the factory. Resolve rank without changing the constructor signature:

```python
rank = (
    torch.distributed.get_rank()
    if torch.distributed.is_available()
    and torch.distributed.is_initialized()
    else 0
)
```

Add manager wrappers that return exact rank/mode receipts, validate context,
delegate to the recorder, and leave mode off by default. `drain` must call the
recorder with `self.synchronize_copies` only after the worker has completed
generation and cleanup; it must never be called from attention or H2D enqueue.

- [x] **Step 3: Write RED tests for every production mapping mutation**

Add tests that configure observe mode and exercise:

1. initial assignment in `ensure_resident`;
2. eviction plus reassignment;
3. contiguous-slot reorder;
4. `_clear_logical_block_metadata`;
5. `discard_resident_blocks`;
6. `evict_clean_resident_blocks`;
7. rollback restoration in `discard_resident_blocks`.

Each test must compare `manager.logical_to_slot` and
`manager.slot_to_logical` with the diagnostic active occupancy inventory.
The reassignment test must assert physical generations advance even when the
same logical block later returns to the slot.

- [x] **Step 4: Centralize mapping-to-diagnostic transitions without changing mapping semantics**

Add focused private helpers:

```python
def _diagnostic_assign_slot(
    self,
    *,
    physical_slot: int,
    logical_block: int,
) -> None:
    diagnostic = self._h2d_slot_reuse_diagnostic
    if not diagnostic.enabled:
        return
    generation = self.bound_generations[logical_block]
    if generation is None:
        raise RuntimeError(
            "diagnostic assignment requires bound generation"
        )
    diagnostic.assign_slot(
        physical_slot=physical_slot,
        logical_block=logical_block,
        bound_generation=generation,
    )


def _diagnostic_release_slot(
    self,
    *,
    physical_slot: int,
    logical_block: int,
) -> None:
    if self._h2d_slot_reuse_diagnostic.enabled:
        self._h2d_slot_reuse_diagnostic.release_slot(
            physical_slot=physical_slot,
            logical_block=logical_block,
        )
```

Call these helpers at every mapping write point identified in
`KVOffloadMVP0`:

- `_clear_logical_block_metadata`;
- `discard_resident_blocks`;
- `evict_clean_resident_blocks`;
- `_evict_slot` when it clears an old mapping;
- the first assignment loop in `ensure_resident`;
- both clear and assignment loops in contiguous-slot reorder;
- rollback restoration after a failed discard.

For replacement inside `ensure_resident`, call `assign_slot` before erasing
the old occupancy so the recorder can snapshot predecessor reads. For a pure
release with no immediate replacement, call `release_slot`. The recorder must
fail if the production mapping and diagnostic occupancy disagree.

- [x] **Step 5: Add a post-mutation consistency assertion used only when enabled**

Implement:

```python
def _assert_h2d_slot_reuse_diagnostic_mapping(self) -> None:
    diagnostic = self._h2d_slot_reuse_diagnostic
    if not diagnostic.enabled:
        return
    expected = tuple(
        None
        if logical_block is None
        else (
            slot,
            int(logical_block),
            int(self.bound_generations[logical_block]),
        )
        for slot, logical_block in enumerate(
            self.slot_to_logical
        )
    )
    diagnostic.assert_mapping(expected)
```

Invoke it after each complete mapping transaction, not between the clear and
set halves of an intentional replacement. This assertion performs no CUDA
work.

- [x] **Step 6: Run manager mapping tests GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_kv_offload.py \
  tools/test_kv_offload_generation_metadata.py
```

Expected: existing and new mapping tests pass; off mode allocates no event.

---

### Task 3: Place Read-Completion Markers After All K/V Slot Reads

**Files:**
- Modify: `tinyvllm/layers/attention.py`
- Modify: `tools/test_blockwise_attention_planning.py`
- Modify: `tools/test_native_verifier_attention.py`

**Interfaces:**
- Consumes:
  - `KVOffloadMVP0.record_h2d_slot_read_window(...)`
- Produces:
  - one marker per staged decode/spec-verify/prefill read window;
  - `attention_stage` exactly `decode`, `spec_verify`, or `prefill`;
  - physical slot and active occupancy captured after the final K/V read is
    enqueued and before later compute consumes the dense buffer.

- [x] **Step 1: Write RED decode marker-placement test**

In `tools/test_blockwise_attention_planning.py`, extend the fake manager with
`record_h2d_slot_read_window()` and append an ordered operation log. Run a
single decode window and assert:

```python
assert operation_log == [
    "stage",
    "k_read:0",
    "v_read:0",
    "k_read:1",
    "v_read:1",
    "read_marker:decode:layer=2:window=0",
]
```

Patch the fake K/V cache accessors rather than relying on CUDA. Also assert the
marker receives exact logical blocks and physical slots in first-seen window
order.

- [x] **Step 2: Implement the decode marker**

In `_blockwise_online_decode_attention()`, enumerate windows:

```python
for window_ordinal, window_plan in enumerate(window_plans):
```

After the nested K/V copy loops and before `k_dense.to(torch.float32)`, call:

```python
manager.record_h2d_slot_read_window(
    engine_step=None,
    attention_stage="decode",
    layer_index=int(layer_idx),
    window_ordinal=window_ordinal,
    logical_blocks=tuple(required_blocks),
    physical_slots=tuple(
        manager.logical_to_slot[int(block)]
        for block in required_blocks
    ),
    current_stream=torch.cuda.current_stream(),
)
```

The manager wrapper must use its previously configured engine-step context
when `engine_step=None`. In off mode it must return before requesting the
current stream; implement the wrapper so CPU tests do not invoke CUDA.

- [x] **Step 3: Write RED spec-verify and prefill marker-placement tests**

In `tools/test_native_verifier_attention.py`, assert one spec-verify marker is
recorded after every K/V slot read and receives `layer_idx`.

In `tools/test_blockwise_attention_planning.py`, assert prefill:

- marks each historical-prefix window;
- does not mark the fresh local causal K/V path;
- labels rows `prefill`;
- records the layer index;
- places the marker before float conversion and score computation.

- [x] **Step 4: Implement shared placement for spec-verify and prefill**

In `_blockwise_online_spec_verify_attention()`, enumerate the actual window
plan used by even/odd layer order and call the same manager wrapper after the
last K/V slot read for each window.

Change the prefill helper signature to:

```python
def _blockwise_online_prefill_attention(
    q,
    k,
    v,
    k_cache,
    v_cache,
    context,
    num_heads,
    head_dim,
    scale,
    layer_idx: int = -1,
):
```

Pass `getattr(self, "layer_idx", -1)` from the `Attention.forward()` prefill
dispatch. Enumerate each historical-prefix window and record after its final
K/V read. Do not mark the local `k`/`v` tensors because they are not reads from
reused physical KV slots.

- [x] **Step 5: Prove off-mode non-invasiveness at the attention boundary**

Add one test per stage whose manager wrapper raises if it requests an event or
CUDA stream while mode is off. Assert outputs remain equal to the existing
dense references and operation logs contain no marker row.

- [x] **Step 6: Run focused attention tests GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_blockwise_attention_planning.py \
  tools/test_native_verifier_attention.py
```

Expected: all existing numerical/reference tests and new placement tests pass.

---

### Task 4: Instrument H2D Spans and Add the Narrow Control Edge

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_kv_offload.py`

**Interfaces:**
- Consumes:
  - pending replacement occupancies from Task 2;
  - `H2DSlotReuseDiagnostic.begin_h2d_span()` and
    `end_h2d_span()`.
- Produces:
  - observe-mode timing rows with zero waits;
  - control-mode unique predecessor waits before H2D start;
  - unchanged production H2D event/counter/pair/span behavior.

- [x] **Step 1: Write RED tests for span deduplication and exact operation order**

Extend `_DummyStream` to log `wait_event`, event `record`, and copy calls.
Build a coalesced span where two destination slots share one prior read event.
Assert:

```python
assert operation_log == [
    "wait:read_event_1",
    "record:h2d_start",
    "copy:logical=4:slot=0:span=2",
    "record:h2d_done",
    "record:production_h2d_done",
]
assert copy_stream.waited == ["read_event_1"]
```

Repeat in observe mode and assert the operation log omits only the wait. Add a
second test with two distinct predecessor events and assert each is waited
exactly once.

- [x] **Step 2: Instrument each existing H2D span without changing the copy**

Inside the existing `with torch.cuda.stream(self.copy_stream):` loop in
`_enqueue_h2d_pairs()`:

```python
copy_batch_ordinal = (
    self._h2d_slot_reuse_copy_batch_ordinal
)
self._h2d_slot_reuse_copy_batch_ordinal += 1
for copy_span_ordinal, (
    logical_start,
    slot_start,
    span_len,
) in enumerate(spans):
    span_pairs = tuple(
        (
            logical_start + offset,
            slot_start + offset,
        )
        for offset in range(span_len)
    )
    diagnostic_handle = (
        self._h2d_slot_reuse_diagnostic.begin_h2d_span(
            copy_batch_ordinal=copy_batch_ordinal,
            copy_span_ordinal=copy_span_ordinal,
            pairs=span_pairs,
            copy_stream=self.copy_stream,
        )
        if self._h2d_slot_reuse_diagnostic.enabled
        else None
    )
    self.kv_cache[
        :,
        :,
        slot_start:slot_start + span_len,
    ].copy_(
        self.cpu_cache[
            :,
            :,
            logical_start:logical_start + span_len,
        ],
        non_blocking=True,
    )
    if diagnostic_handle is not None:
        self._h2d_slot_reuse_diagnostic.end_h2d_span(
            diagnostic_handle,
            copy_stream=self.copy_stream,
        )
```

Keep the existing logical-block D2H waits before diagnostic predecessor waits.
Keep `_record_copy_event()` after all spans exactly as the production
completion event for `h2d_done`. Do not substitute the diagnostic done event.

- [x] **Step 3: Fail closed for synchronous-copy mode while enabled**

The focused matrix requires `async_copy=true`. If the diagnostic is enabled
and `self.copy_stream is None`, `_enqueue_h2d_pairs()` must raise:

```text
H2D slot-reuse diagnostic requires asynchronous copy stream
```

Mode off must preserve the existing synchronous-copy path unchanged.

- [x] **Step 4: Prove pairs, spans, counters, bytes, and production events are unchanged**

Write a parameterized test that executes identical H2D pairs under `off`,
`observe`, and `control`, then compares:

- `_coalesce_copy_pairs()` output;
- `h2d_copies`;
- `h2d_bytes`;
- `h2d_batches`;
- `h2d_batch_spans`;
- `pending_wait_blocks`;
- logical-block keys in `h2d_done`;
- production event sharing across coalesced logical blocks.

Only diagnostic wait count and event rows may differ. Also compare D2H
counters/events before and after to prove no new D2H work.

- [x] **Step 5: Test no-prior cases and failure cleanup**

Add manager-level tests for:

- H2D into a never-occupied slot -> `NO_PRIOR_OCCUPANCY`;
- H2D replacing an occupancy never read -> `NO_PRIOR_READ`;
- exception during copy -> manager lifecycle is disabled/cleared by the
  worker-facing `finally`, while production exceptions still propagate;
- invalid pending replacement identity -> hard error before copy;
- control adds no current-stream wait and no global synchronize.

- [x] **Step 6: Run KV-offload integration tests GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_h2d_slot_reuse_diagnostic.py \
  tools/test_kv_offload.py \
  tools/test_kv_offload_generation_metadata.py
```

Expected: all tests pass and the existing production completion-event tests
remain unchanged.

---

### Task 5: Add All-Rank Engine Lifecycle and Evidence Collection

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Create:
  `tools/test_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py`

**Interfaces:**
- Produces on `ModelRunner`:
  - `configure_h2d_slot_reuse_diagnostic(mode) -> dict`
  - `set_h2d_slot_reuse_diagnostic_context(engine_step, attention_stage) -> dict`
  - `drain_h2d_slot_reuse_diagnostic(timing_epsilon_ms) -> dict`
  - `h2d_slot_reuse_diagnostic_summary() -> dict`
- Produces on `LLMEngine`:
  - same lifecycle across all ranks with `timeout_s`;
  - rank-sorted receipts and drain rows.
- Consumed by the focused worker only.

- [x] **Step 1: Write RED acknowledged-command tests**

Create fake local and worker results and assert:

```python
receipt = engine.configure_h2d_slot_reuse_diagnostic(
    "observe",
    timeout_s=60.0,
)
assert receipt == {
    "mode": "observe",
    "rank_inventory": [0, 1, 2, 3],
}
```

Add failures for:

- missing rank;
- duplicate rank;
- inconsistent mode;
- malformed drain schema;
- non-sorted rank rows;
- a worker returning `mode="off"` while observe is requested.

- [x] **Step 2: Implement exact model-runner receipts**

`ModelRunner.configure_h2d_slot_reuse_diagnostic()` must reject an absent
`self.kv_offload` manager, then return exactly:

```python
{
    "rank": self.rank,
    "mode": mode,
}
```

The context method must accept an integer `engine_step` and the ordinary
baseline stages `attention_stage in {"prefill", "decode"}` for the focused
campaign. It must reject `spec_verify` at this wrapper because the focused
worker is baseline-only. Keep the lower-level manager capable of all three
stages so direct attention tests remain valid.

The drain method must call the manager only after the worker synchronizes
generation/cleanup, and return:

```python
{
    "rank": self.rank,
    "schema": H2D_SLOT_REUSE_SCHEMA,
    "mode": mode,
    "stream_inventory": [...],
    "read_rows": [...],
    "overwrite_rows": [...],
}
```

- [x] **Step 3: Implement all-rank `LLMEngine` wrappers**

Use `call_model_runner_acknowledged()` following
`enable_step_logits_authority_recording()`:

```python
def configure_h2d_slot_reuse_diagnostic(
    self,
    mode,
    *,
    timeout_s,
):
    local, acks = self.call_model_runner_acknowledged(
        "configure_h2d_slot_reuse_diagnostic",
        mode,
        timeout_s=timeout_s,
    )
    rows = [local, *(ack.result for ack in acks)]
    return _validate_rank_mode_rows(
        rows,
        world_size=self.model_runner.world_size,
        expected_mode=mode,
    )
```

Add corresponding context, drain, and summary methods. The drain wrapper must
return a tuple sorted by `rank`, reject missing/duplicate ranks, and reject
schema/mode mismatches.

- [x] **Step 4: Test failure-safe clearing on every rank**

Use a fake engine where rank 2 drain fails. The worker-facing lifecycle helper
must still dispatch `configure(..., "off")` to all four ranks in `finally`.
Assert no rank remains enabled and the original exception is re-raised.

- [x] **Step 5: Run engine lifecycle tests GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py \
  tools/test_kv_offload.py
```

Expected: all-rank lifecycle receipts are exact and cleanup is unconditional.

---

### Task 6: Build the Focused Baseline-Only Worker and Artifact

**Files:**
- Create:
  `tools/qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_gate.py`
- Create:
  `tools/qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_worker.py`
- Modify:
  `tools/test_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py`

**Interfaces:**
- Gate produces frozen constants, field validators, source manifest, numeric
  logit comparison, movement comparison, and artifact validation.
- Worker produces exactly four logical cells:
  - `observe:b1`
  - `observe:b4`
  - `control:b1`
  - `control:b4`
- Worker records rank-zero compact logits for prediction indices 0 and 1 and
  rank-local slot rows from all four ranks.

- [x] **Step 1: Write RED tests for frozen constants and baseline-only keys**

Assert:

```python
assert gate.SCHEMA == (
    "qwen35.tp4-32k-h2d-slot-reuse-causal-diagnostic.v1"
)
assert gate.MODES == ("observe", "control")
assert gate.BATCH_SIZES == (1, 4)
assert gate.REQUIRED_CELL_KEYS == (
    "observe:b1",
    "observe:b4",
    "control:b1",
    "control:b4",
)
assert gate.POLICY == "baseline"
assert gate.PROMPT_TOKENS == 32768
assert gate.MAX_OUTPUT_TOKENS == 8
assert gate.MAX_PROPOSAL_TOKENS == 4
assert gate.WORLD_SIZE == 4
assert gate.BLOCK_SIZE == 256
assert gate.GPU_BLOCKS == 68
assert gate.LOGICAL_BLOCKS == 640
assert gate.BLOCKWISE_BLOCKS == 8
assert gate.TIMING_EPSILON_MS == 0.2
assert gate.TOP_K == 5
```

Test rejection of `native_mtp:b1`, `baseline:b1`, missing cells, duplicate
repetition IDs, and one-cell-only artifacts.

- [x] **Step 2: Implement the focused gate without importing paired trace helpers**

The gate may load the frozen 32K target-KV gate for model/source constants and
shared JSON/SHA helpers, but must define its own diagnostic schema and exact
artifact validator. Its source manifest must include:

```text
tinyvllm/engine/h2d_slot_reuse_diagnostic.py
tinyvllm/engine/model_runner.py
tinyvllm/layers/attention.py
tinyvllm/engine/llm_engine.py
tools/qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_gate.py
tools/qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_worker.py
tools/verify_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py
```

Do not import
`tools/qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py`; that module owns
paired-trace code and is outside this focused lifecycle.

- [x] **Step 3: Write RED tests for compact prediction-index 0/1 logits**

Use a fake two-step ordinary baseline engine. Assert the worker:

- enables only step-logit recording;
- never calls `enable_spec_verify_trace_recording`;
- never touches `qwen35_speculative_state_owner`;
- records exactly prediction indices 0 and 1 per prompt;
- uses deterministic descending-logit/ascending-token-ID ordering;
- records `top_k=5`, `input_token_id`, `position`, `context_length`,
  `top1_margin`, and `argmax_token`;
- ignores later prediction indices for compact-logit evidence while still
  generating all eight output tokens.

- [x] **Step 4: Implement focused generation lifecycle**

Implement a helper equivalent to:

```python
def run_generation_with_h2d_slot_reuse_diagnostic(
    *,
    engine,
    prompt_rows,
    sampling_params,
    synchronize,
    mode,
    batch_size,
    repetition,
    timing_epsilon_ms,
    target_forward_capture=None,
):
    engine.configure_h2d_slot_reuse_diagnostic(
        mode,
        timeout_s=60.0,
    )
    engine.enable_step_logits_authority_recording(
        True,
        timeout_s=60.0,
    )
    outputs_by_id = {}
    compact_logits = []
    observations = []
    try:
        for row in prompt_rows:
            engine.add_request(
                row["token_ids"],
                sampling_params,
            )
        engine_step = 0
        while not engine.is_finished():
            engine.set_h2d_slot_reuse_diagnostic_context(
                engine_step,
                "decode",
                timeout_s=60.0,
            )
            before = _ordinary_forward_count(
                target_forward_capture
            )
            step_outputs, _ = engine.step()
            synchronize()
            after = _ordinary_forward_count(
                target_forward_capture
            )
            observation = _validated_baseline_observation(
                engine.last_step_observation,
                before=before,
                after=after,
            )
            if engine_step in (0, 1):
                compact_logits.extend(
                    compact_prediction_logits(
                        engine.read_step_logits_authority(),
                        observation=observation,
                        prediction_index=engine_step,
                        top_k=5,
                    )
                )
            observations.append(observation)
            for sequence_id, token_ids in step_outputs:
                outputs_by_id[int(sequence_id)] = [
                    int(token_id) for token_id in token_ids
                ]
            engine_step += 1
        engine.flush_pending_hybrid_state_releases(
            timeout_s=60.0,
        )
        slot_rows = engine.drain_h2d_slot_reuse_diagnostic(
            timing_epsilon_ms,
            timeout_s=60.0,
        )
    finally:
        engine.enable_step_logits_authority_recording(
            False,
            timeout_s=60.0,
        )
        engine.configure_h2d_slot_reuse_diagnostic(
            "off",
            timeout_s=60.0,
        )
```

The real implementation must preserve the original exception if cleanup also
fails, while attaching cleanup failures as notes where supported. No drain is
allowed before pending copies and generation cleanup finish.

- [x] **Step 5: Reuse the frozen 32K ordinary cell without native runtime activation**

Load
`qwen35_native_mtp_tp4_16k_target_kv_offload_worker.py`, inject the frozen 32K
gate constants exactly as the existing 32K authority wrapper does, and call
`run_policy_cell(policy="baseline", ...)` with the focused generation
function. Reject any policy argument other than `baseline` before engine
construction.

For each cell, preserve:

- frozen model and checkpoint manifest validation;
- frozen prompt builder;
- exact greedy sampling;
- pre/post `kv_offload_summaries`;
- `kv_rank_deltas`;
- capacity rows;
- rank snapshots;
- cleanup receipts;
- source-tree digest;
- target checkpoint digest;
- ordinary target-forward count.

- [x] **Step 6: Record required runtime metadata and fail closed**

The cell metadata must include:

```text
torch_version
torch_cuda_runtime_version
nvidia_driver_version
cuda_device_names
```

Use `torch.__version__`, `torch.version.cuda`, NVML or
`nvidia-smi --query-gpu=driver_version --format=csv,noheader`, and
`torch.cuda.get_device_name()`. Dependency injection must make local tests
independent of an installed driver. Empty PyTorch version, CUDA runtime
version, driver version, or device inventory is a hard diagnostic failure.

- [x] **Step 7: Assemble tensor-free per-repetition artifacts**

Each repetition must contain exact fields:

```text
schema
mode
policy
batch_size
repetition
world_size
prompt_tokens
max_output_tokens
max_proposal_tokens
block_size
gpu_blocks
logical_blocks
blockwise_blocks
async_copy
batch_copy
writeback_on_evict
enforce_eager
torch_version
torch_cuda_runtime_version
nvidia_driver_version
cuda_device_names
source_tree_sha256
checkpoint_sha256
timing_epsilon_ms
prompt_rows
output_rows
compact_logit_rows
rank_slot_rows
step_observations
target_forward_count
kv_rank_deltas
kv_capacity_rows
cleanup
cell_digest_sha256
```

Reject tensors recursively before writing JSON. Store failed artifacts with a
failure status and error string; never publish a terminal causal
classification from an incomplete rank/cell.

- [x] **Step 8: Run worker/gate tests GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py
```

Expected: baseline-only lifecycle, exact metadata, compact logits, all-rank
rows, and cleanup tests pass without constructing a real GPU engine.

---

### Task 7: Implement Invariants and the Mutually Exclusive Decision Matrix

**Files:**
- Modify:
  `tools/qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_gate.py`
- Create:
  `tools/verify_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py`
- Modify:
  `tools/test_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py`

**Interfaces:**
- Produces terminal classification exactly one of:
  - `TP4_32K_H2D_SLOT_REUSE_GPU_CAUSALITY=SUPPORTED`
  - `TP4_32K_H2D_SLOT_REUSE_GPU_CAUSALITY=REJECTED`
  - `TP4_32K_H2D_SLOT_REUSE_GPU_CAUSALITY=INCONCLUSIVE`
- Local dummy evidence must always remain `INCONCLUSIVE`; only a future real
  GPU artifact can support or reject the hypothesis.

- [x] **Step 1: Write RED invariant tests**

Create one valid synthetic campaign with two repetitions per cell, then
mutate one field at a time and require failure for:

- prompt token IDs or sequence order;
- source-tree/checkpoint identity;
- engine/KV configuration;
- output length;
- target-forward count;
- logical block identity inventory;
- H2D/D2H copies or bytes;
- H2D/D2H batches or spans;
- eviction count;
- peak resident blocks;
- cleanup inventory;
- world/rank inventory;
- missing PyTorch/CUDA/driver/device metadata;
- native-MTP stage;
- paired-trace row;
- side-state row;
- proposal callback;
- shadow or extra target forward.

Allow differences only in timestamps, diagnostic wait count, wall time,
compact logits, and output tokens.

- [x] **Step 2: Implement exact observe/control comparison**

For each batch size and repetition, compare a normalized invariant projection:

```python
INVARIANT_FIELDS = (
    "prompt_rows",
    "source_tree_sha256",
    "checkpoint_sha256",
    "world_size",
    "prompt_tokens",
    "max_output_tokens",
    "max_proposal_tokens",
    "block_size",
    "gpu_blocks",
    "logical_blocks",
    "blockwise_blocks",
    "async_copy",
    "batch_copy",
    "writeback_on_evict",
    "enforce_eager",
    "target_forward_count",
    "kv_rank_deltas",
    "kv_capacity_rows",
    "cleanup",
)
```

Normalize rank rows by rank and reject any movement-inventory difference
before evaluating logits or overlap. Also compare exact H2D/D2H pair/span
inventories serialized by the manager; aggregate counters alone are
insufficient.

- [x] **Step 3: Write RED timing-coverage and control-effect tests**

Require:

- observe has at least one valid `UNSAFE_OVERLAP_OBSERVED` row to be eligible
  for `SUPPORTED`;
- control has zero unsafe rows and a matching waited predecessor for every
  observed hazardous physical-slot occupancy;
- ambiguous/missing timing cannot count safe;
- `NO_PRIOR_OCCUPANCY` and `NO_PRIOR_READ` remain inventory rows but cannot
  support or reject the hazard;
- all four ranks have complete stream/event lifecycles;
- unsupported extra stream representation is inconclusive.

- [x] **Step 4: Implement predeclared prediction-index-1 drift comparison**

Use a fixed numeric rule declared in the gate:

```python
LOGIT_ATOL = 1e-5
LOGIT_RTOL = 1e-5
```

Compare `b1` prompt 0 with `b4` prompt 0 at prediction index 1:

- first require exact prompt-0 token identity across `observe:b1`,
  `observe:b4`, `control:b1`, and `control:b4`;
- first require exact prediction-index-0/1 semantic identity across all four
  cells using `(prediction_index, input_token_id, position, context_length)`,
  with `context_length == position + 1`;
- drift reproduced when shared top-token ordering differs, argmax differs, or
  any shared top-5 token logit violates `atol + rtol * abs(reference)`;
- drift removed only when top-token ordering and argmax match and every shared
  top-5 logit is within tolerance;
- missing or mismatched prompt/prediction identity is inconclusive;
- missing index-1 row is inconclusive;
- final token equality without compact-logit equality is inconclusive.

Prediction index 0 remains a retained control row and must be reported, but
index 1 is the dependent variable for the decision.

- [x] **Step 5: Write RED supported/rejected/inconclusive matrix tests**

Build fixtures for:

1. **Supported**
   - observe reproduces drift and records unsafe overlap;
   - control waits matching predecessors, records zero unsafe overlap, removes
     index-1 drift, and preserves exact greedy output parity;
   - all invariants pass.
2. **Rejected**
   - observe reproduces drift with complete valid timing but no unsafe overlap;
     or
   - control removes overlap while drift/output mismatch remains unchanged.
3. **Inconclusive**
   - observe does not reproduce drift;
   - timing incomplete/ambiguous;
   - prompt-0 token identity differs across cells;
   - prediction-index-0/1 semantic identity differs across cells;
   - identity or movement differs;
   - control changes outputs without matching overlap;
   - only final tokens agree;
   - one rank has incomplete lifecycle;
   - only one repetition is present.

Assert exactly one terminal boolean is true for every fixture.

- [x] **Step 6: Implement terminal decision evaluation**

Return:

```python
{
    "classification": classification,
    "supported": classification.endswith("=SUPPORTED"),
    "rejected": classification.endswith("=REJECTED"),
    "inconclusive": classification.endswith("=INCONCLUSIVE"),
    "reasons": [...],
    "repetition_inventory": {
        "observe:b1": [...],
        "observe:b4": [...],
        "control:b1": [...],
        "control:b4": [...],
    },
}
```

Reject a result if the booleans are not mutually exclusive. A single
repetition per cell must classify as inconclusive even when the observed
pattern matches the supported matrix.

- [x] **Step 7: Implement independent verifier CLI**

The verifier must:

1. load the artifact and source manifest;
2. recompute source-tree SHA-256 from the copied source;
3. validate target checkpoint identity;
4. validate every cell and repetition;
5. evaluate invariant and decision matrices;
6. print one compact JSON result;
7. exit nonzero for malformed/tampered evidence;
8. retain `INCONCLUSIVE` as a valid completed campaign outcome.

CLI shape:

```bash
python3 tools/verify_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py \
  --run-dir /path/to/run \
  --repo-root /path/to/copied/source \
  --model /path/to/frozen/checkpoint
```

- [x] **Step 8: Run decision/verifier tests GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py
```

Expected: all supported/rejected/inconclusive fixtures are exclusive, and
tampering fails closed.

---

### Task 8: Run the Complete Local Contract Gate and Update the Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Verify all files listed above.

**Interfaces:**
- Produces local-only evidence for:
  - `H2D_SLOT_REUSE_DIAGNOSTIC_CONTRACT=ESTABLISHED`
  - `DEFAULT_OFF_NON_INVASIVENESS=ESTABLISHED`
- Does not produce GPU causality, production-fix, TP4/32K correctness, or
  Phase 1 promotion evidence.

- [x] **Step 1: Run syntax compilation**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/h2d_slot_reuse_diagnostic.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/layers/attention.py \
  tinyvllm/engine/llm_engine.py \
  tools/qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_gate.py \
  tools/qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_worker.py \
  tools/verify_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py
```

Expected: exit code 0.

- [ ] **Step 2: Run the focused local suite**

Run:

```bash
python3 -m pytest -q \
  tools/test_h2d_slot_reuse_diagnostic.py \
  tools/test_kv_offload.py \
  tools/test_kv_offload_generation_metadata.py \
  tools/test_blockwise_attention_planning.py \
  tools/test_native_verifier_attention.py \
  tools/test_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py
```

Expected: all tests pass.

Current 2026-08-15 boundary: the dependency-light focused replacements pass,
but `tools/test_kv_offload.py` and
`tools/test_blockwise_attention_planning.py` cannot collect on this host
because production imports require `flash_attn` and `triton`. This step stays
unchecked until it runs in the approved CUDA/FlashAttention environment; the
collection blocker is not counted as a test pass or failure.

- [x] **Step 3: Run adjacent regression suites**

Run:

```bash
python3 -m pytest -q \
  tools/test_model_runner_spec_verify.py \
  tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py
```

Expected: all tests pass; paired trace and authority schemas remain unchanged.

- [x] **Step 4: Run source-level non-invasiveness checks**

Run:

```bash
rg -n \
  "enable_spec_verify_trace_recording|enable_trace_recording|native_mtp|shadow_forward" \
  tools/qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_worker.py
```

Expected: only explicit rejection/absence assertions appear; no activation
call exists.

Run:

```bash
rg -n \
  "torch\\.cuda\\.synchronize|copy_stream\\.synchronize|dist\\.barrier|wait_stream" \
  tinyvllm/engine/h2d_slot_reuse_diagnostic.py \
  tinyvllm/engine/model_runner.py
```

Expected: no new diagnostic hot-path synchronization; existing production
uses are reviewed by diff rather than treated as new.

- [x] **Step 5: Review scoped diff and whitespace**

Run:

```bash
git diff --check -- \
  tinyvllm/engine/h2d_slot_reuse_diagnostic.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/layers/attention.py \
  tinyvllm/engine/llm_engine.py \
  tools/qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_gate.py \
  tools/qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_worker.py \
  tools/verify_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py \
  tools/test_h2d_slot_reuse_diagnostic.py \
  tools/test_kv_offload.py \
  tools/test_blockwise_attention_planning.py \
  tools/test_native_verifier_attention.py \
  tools/test_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py \
  AGENT_HANDOFF_STATE.md
```

Expected: no whitespace errors.

- [x] **Step 6: Perform a prompt-to-artifact completion audit**

Record a checklist in the handoff mapping each approved requirement to
evidence:

```text
physical slot + occupancy generation
all mapping transitions covered
decode/spec-verify/prefill read marker placement
observe records without waits
control waits unique predecessors only
diagnostic events separate from h2d_done/d2h_done
fixed timing epsilon and classification
immutable tensor-free drain
all-rank rows
baseline-only b1/b4 cells
prediction-index 0/1 top-5 logits
PyTorch/CUDA/driver/device metadata
movement and target-forward invariants
supported/rejected/inconclusive exclusivity
default-off non-invasiveness
failure cleanup
no GPU run
no production fix
claim boundary preserved
```

Treat any missing test or artifact mapping as incomplete and continue locally
instead of claiming contract establishment.

- [x] **Step 7: Update `AGENT_HANDOFF_STATE.md`**

Append:

- exact implementation files;
- exact commands and pass counts;
- any environment-limited tests;
- whether source-level non-invasiveness checks passed;
- local classifications established;
- unchanged GPU/root-cause/production/Phase-1 boundaries;
- the next action: request separate authorization for the four-cell real
  TP4/32K campaign, including repetition count and remote route.

- [x] **Step 8: Stop before any GPU or remote action**

Do not create or run a remote script, reserve GPUs, probe the remote host,
launch NCCL, or modify an authority artifact. Report that implementation is
locally ready and ask for a separate focused GPU campaign authorization.

## 2026-08-15 Fresh Final Readiness Evidence

The implementation and local evidence were revalidated without GPU, remote,
NCCL, loaded-checkpoint, or performance execution:

```text
focused diagnostic, manager, attention-marker, worker, gate, and verifier:
  99 passed in 0.51s

ModelRunner spec-verify and native-MTP TP4/16K adjacent regressions:
  169 passed in 4.10s

native-MTP TP4/32K authority gate contract:
  35 passed in 8.84s

seven implementation/producer/verifier files py_compile:
  PASS

baseline-only four-cell and seven-file manifest static contract:
  PASS

focused worker GPU/remote launcher scan:
  PASS

source-level non-invasiveness review:
  PASS

scoped git diff --check:
  PASS
```

Plan status:

```text
Tasks 1-7:                         COMPLETE
Task 8 steps 1,3,4,5,6,7,8:      COMPLETE
Task 8 step 2 full CUDA suite:    COLLECTION_BLOCKED
```

The blocked files require the real production CUDA extension import
environment. No fake `flash_attn`, Triton, or CUDA module is accepted as
evidence.

```text
FOCUSED_H2D_DIAGNOSTIC_LOCAL_CONTRACT=ESTABLISHED
FOCUSED_H2D_FULL_PLAN_TEST_MATRIX=PARTIAL_ENVIRONMENT_BLOCKED
TP4_32K_H2D_SLOT_REUSE_GPU_CAUSALITY=NOT_ESTABLISHED
PRODUCTION_H2D_SLOT_REUSE_FIX=NOT_APPROVED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

