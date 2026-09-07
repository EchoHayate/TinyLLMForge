# TP4 Segmented Capture Attribution Phase A1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one immutable, independently verified Phase A1 attribution bundle that localizes the alternating TP4 segmented-capture latency and the first scratch-KV divergence boundary, then terminates as `REPAIR_CANDIDATE`, `PIVOT_TP4_COMMUNICATION_COMPUTE_FUSION`, or `INCOMPLETE`.

**Architecture:** Keep the completed r60 census implementation and artifacts unchanged. Add a dependency-light attribution contract, a separate diagnostic worker with S0-S7 scratch checkpoints and phase timing, and a separate strict-clean controller/verifier pair that reuses the proven source-freeze, Kerberos, GPU-admission, lifecycle, archive, and exact-tag cleanup primitives without changing their gates.

**Tech Stack:** Python 3.12, PyTorch CUDA Graphs, torch.distributed/NCCL, pytest, JSON/JSONL evidence, SHA-256 manifests, SSH/Kerberos remote controller, Qwen3.8 hybrid-state runtime.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge`.
- Do not update `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not create a worktree or dispatch subagents; execute this plan inline.
- Push only to `origin/feat/kv-sparse-attention`.
- Use strict RED -> minimal implementation -> GREEN for every runtime change.
- Stage exact paths only; never use `git add -A`, `git reset`, `git clean`, or broad formatting.
- Commit with `git -c core.hooksPath=/dev/null commit`.
- Every commit must contain exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Preserve all unrelated tracked and untracked files.
- Keep the completed r57-r60 source, artifacts, manifests, and classifications immutable.
- Do not integrate diagnostic code into production graph dispatch or create production graph-cache entries.
- This plan implements Phase A1 only. It must not implement a hypothetical Phase A2 repair.
- Stop Phase A1 at exactly one of:
  - `REPAIR_CANDIDATE`;
  - `PIVOT_TP4_COMMUNICATION_COMPUTE_FUSION`;
  - `INCOMPLETE`.
- `GO_SEGMENTED_REPAIR` is invalid for a Phase A1 bundle.
- Keep the frozen model and workload:
  - repository `Qwen/Qwen3.8-27B`;
  - revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
  - BF16;
  - tensor parallel size 4;
  - batch/concurrency 8;
  - prompt length 256;
  - worker max tokens 2;
  - model length 384.
- Keep the frozen timing limits:
  - maximum segment `1_800_000_000 ns`;
  - complete lifecycle `4_500_000_000 ns`.
- Do not move snapshot, preparation, graph creation, capture entry/body/exit, synchronization, restore, memory sampling, or reset work outside measured lifecycle accounting.
- Preserve exact output, selected-state, unselected-state, graph-reset, memory, source, admission, lifecycle, and cleanup checks.
- Formal evidence requires `strict_clean`; `shared_capacity` is diagnostic only and cannot select a route.
- Do not execute `kinit` or `krenew`.
- Never terminate, suspend, adopt, or clean foreign GPU processes.
- Exact-tag-owned cleanup is mandatory and may reuse the existing lifecycle owner.
- Keep all remote source, cache, logs, artifacts, and temporary files below `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- Never write remote experiment data under `/`.
- Do not copy model weights, caches, full scratch tensors, or other large artifacts to the Mac.
- Prior run tags and evidence directories are immutable.
- A disconnect does not authorize a duplicate launch; inspect supervisor, process, and artifact state first.
- Report both benefit and cost. Attribution is not a steady-state performance claim.

---

## Phase A1 Boundary

This plan implements only the diagnostic source and one strict-clean Phase A1
run. It does not guess the eventual source repair.

The terminal handoff is:

```text
REPAIR_CANDIDATE
  -> write a new repair design and implementation plan from the immutable A1
     evidence; permit exactly one fresh-source Phase A2 run

PIVOT_TP4_COMMUNICATION_COMPUTE_FUSION
  -> close segmented capture and write a separate steady-state TP4
     communication-compute fusion design

INCOMPLETE
  -> correct only the evidence/infrastructure defect and use a fresh tag;
     do not choose a technical route
```

The worker may collect measurements that later support a repair, but no task
in this plan changes `model_runner.py`, Qwen3.8 model execution, production
graph caching, or snapshot/restore semantics.

## File and Responsibility Map

- `tinyvllm/engine/segmented_capture_attribution.py`
  - owns dependency-light Phase A1 schemas, phase accounting, scratch digest
    and diff records, rank aggregation, bounded-control validation, and the
    deterministic route classifier.
- `tools/test_segmented_capture_attribution.py`
  - tests pure contracts without CUDA or model weights.
- `tools/tp4_segmented_capture_attribution_worker.py`
  - owns the diagnostic-only CUDA backend, deterministic nonzero scratch
    sentinel, S0-S7 checkpoint capture, stitched/isolated/pool controls,
    phase timestamps, per-rank rows, and worker bundle files.
- `tools/test_tp4_segmented_capture_attribution_worker.py`
  - tests checkpoint order, explicit synchronization, error precedence,
    reverse graph reset, bounded matrix construction, and non-root logits.
- `tools/run_tp4_segmented_capture_attribution.py`
  - owns fresh-tag validation, committed-source freezing, mounted-storage
    preflight, Kerberos TTL guard, four-GPU `strict_clean` admission, one
    launch, download, dual verification, manifests, and exact-tag cleanup.
- `tools/test_run_tp4_segmented_capture_attribution.py`
  - tests the controller pipeline and path/source/admission/lifecycle
    invariants.
- `tools/verify_tp4_segmented_capture_attribution.py`
  - independently reconstructs source/workload identity, phase accounting,
    scratch transitions, control identities, TP-wide maxima, cleanup, and the
    Phase A1 terminal classification.
- `tools/test_verify_tp4_segmented_capture_attribution.py`
  - tests tamper detection, fail-closed behavior, rank disagreement, route
    classification, and manifest binding.
- `docs/superpowers/audits/2026-08-31-tp4-collective-stable-decode-replay-audit.md`
  - receives the immutable Phase A1 result, benefit/cost table, claim
    boundary, and next-route decision.
- `AGENT_HANDOFF_STATE.md`
  - receives the exact commit, run tag, artifact paths, verifier hashes,
    cleanup state, terminal classifier, and immediate next action.

The new controller and worker may import stable helper functions from the
existing census files, but must not modify the r60 schemas or reinterpret old
artifacts.

---

### Task 1: Add pure Phase A1 attribution contracts

**Files:**
- Create: `tinyvllm/engine/segmented_capture_attribution.py`
- Create: `tools/test_segmented_capture_attribution.py`

**Interfaces:**
- Produces: `CAPTURE_PHASE_NAMES: tuple[str, ...]`
- Produces: `SCRATCH_CHECKPOINTS: tuple[str, ...]`
- Produces: `CONTROL_IDS: tuple[str, ...]`
- Produces: `CapturePhaseAccounting`
- Produces: `ScratchTensorDigest`
- Produces: `ScratchDiffSummary`
- Produces: `ScratchCheckpointRecord`
- Produces: `AttributionDiagnosis`
- Produces: `canonical_sha256(value: object) -> str`
- Produces: `validate_checkpoint_sequence(records: tuple[ScratchCheckpointRecord, ...]) -> None`
- Produces: `aggregate_tp4_phase_rows(rows: list[dict]) -> dict`
- Produces: `classify_phase_a1(evidence: dict) -> dict`

- [ ] **Step 1: Write failing phase-accounting tests**

Create `tools/test_segmented_capture_attribution.py` with tests beginning:

```python
from dataclasses import replace

import pytest

from tinyvllm.engine.segmented_capture_attribution import (
    AttributionDiagnosis,
    CapturePhaseAccounting,
    ScratchCheckpointRecord,
    ScratchDiffSummary,
    ScratchTensorDigest,
    aggregate_tp4_phase_rows,
    canonical_sha256,
    classify_phase_a1,
    validate_checkpoint_sequence,
)


def valid_phases():
    return CapturePhaseAccounting(
        snapshot_and_prepare_ns=10,
        graph_object_create_ns=11,
        capture_context_enter_ns=12,
        capture_body_ns=13,
        capture_context_exit_and_instantiate_ns=14,
        post_capture_synchronize_ns=15,
        post_capture_restore_ns=16,
        graph_reset_ns=17,
        segment_total_ns=100,
        program_lifecycle_ns=400,
    )


def test_phase_accounting_requires_every_non_negative_interval():
    assert valid_phases().measured_capture_ns == 65
    with pytest.raises(ValueError, match="non-negative"):
        replace(valid_phases(), capture_body_ns=-1)
    with pytest.raises(ValueError, match="segment_total"):
        replace(valid_phases(), segment_total_ns=64)
    with pytest.raises(ValueError, match="program_lifecycle"):
        replace(valid_phases(), program_lifecycle_ns=99)
```

Also assert that booleans are rejected as integers and that
`segment_total_ns` may include diagnostic overhead but cannot be less than:

```text
graph_object_create_ns
+ capture_context_enter_ns
+ capture_body_ns
+ capture_context_exit_and_instantiate_ns
+ post_capture_synchronize_ns
```

- [ ] **Step 2: Run the phase-accounting test and verify RED**

Run:

```bash
pytest -q tools/test_segmented_capture_attribution.py
```

Expected: collection fails with
`ModuleNotFoundError: tinyvllm.engine.segmented_capture_attribution`.

- [ ] **Step 3: Implement the minimal phase contract**

Create `tinyvllm/engine/segmented_capture_attribution.py` with:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


CAPTURE_PHASE_NAMES = (
    "snapshot_and_prepare_ns",
    "graph_object_create_ns",
    "capture_context_enter_ns",
    "capture_body_ns",
    "capture_context_exit_and_instantiate_ns",
    "post_capture_synchronize_ns",
    "post_capture_restore_ns",
    "graph_reset_ns",
)
SCRATCH_CHECKPOINTS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")
CONTROL_IDS = (
    "stitched_p4_repeat_0",
    "stitched_p4_repeat_1",
    "isolated_0_16",
    "isolated_16_32",
    "isolated_32_48",
    "isolated_48_64",
    "pool_fastest_shared",
    "pool_fastest_isolated",
    "pool_slowest_shared",
    "pool_slowest_isolated",
)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CapturePhaseAccounting:
    snapshot_and_prepare_ns: int
    graph_object_create_ns: int
    capture_context_enter_ns: int
    capture_body_ns: int
    capture_context_exit_and_instantiate_ns: int
    post_capture_synchronize_ns: int
    post_capture_restore_ns: int
    graph_reset_ns: int
    segment_total_ns: int
    program_lifecycle_ns: int

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in values.values()
        ):
            raise ValueError("phase durations must be non-negative integers")
        if self.segment_total_ns < self.measured_capture_ns:
            raise ValueError("segment_total_ns is below measured capture")
        if self.program_lifecycle_ns < self.segment_total_ns:
            raise ValueError("program_lifecycle_ns is below segment_total_ns")

    @property
    def measured_capture_ns(self) -> int:
        return sum(
            getattr(self, name)
            for name in CAPTURE_PHASE_NAMES[1:6]
        )
```

Do not include `snapshot_and_prepare_ns`, `post_capture_restore_ns`, or
`graph_reset_ns` in `measured_capture_ns`; they remain in lifecycle and
segment-total accounting rather than being relabeled as CUDA capture.

- [ ] **Step 4: Add failing scratch-record tests**

Append tests that construct separate K and V digests, verify canonical hash
stability, require dtype/shape/byte count, and reject full tensor payloads:

```python
def digest(sha="a" * 64):
    return ScratchTensorDigest(
        selector="key",
        dtype="torch.bfloat16",
        shape=(64, 8, 1, 4, 128),
        byte_count=1_048_576,
        sha256=sha,
    )


def exact_diff():
    return ScratchDiffSummary(
        equal_to_s0=True,
        mismatching_element_count=0,
        first_mismatch=None,
        max_absolute_difference=0.0,
    )


def test_scratch_checkpoint_sequence_is_exact_and_ordered():
    records = tuple(
        ScratchCheckpointRecord(
            checkpoint=name,
            rank=0,
            synchronized=True,
            keys=digest(),
            values=replace(digest(), selector="value", sha256="b" * 64),
            key_diff=exact_diff(),
            value_diff=exact_diff(),
        )
        for name in ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")
    )
    validate_checkpoint_sequence(records)
    with pytest.raises(ValueError, match="checkpoint order"):
        validate_checkpoint_sequence(records[:-1])
```

Add a mismatch fixture whose first mismatch is:

```python
{
    "layer": 17,
    "scratch_slot_ordinal": 3,
    "head": 2,
    "element_offset": 11,
}
```

Assert that the record contains no tensor bytes and that a non-exact diff
requires a positive mismatch count and a first mismatch location.

- [ ] **Step 5: Implement scratch digest, diff, and sequence contracts**

Add:

```python
@dataclass(frozen=True)
class ScratchTensorDigest:
    selector: str
    dtype: str
    shape: tuple[int, ...]
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class ScratchDiffSummary:
    equal_to_s0: bool
    mismatching_element_count: int
    first_mismatch: dict[str, int] | None
    max_absolute_difference: float


@dataclass(frozen=True)
class ScratchCheckpointRecord:
    checkpoint: str
    rank: int
    synchronized: bool
    keys: ScratchTensorDigest
    values: ScratchTensorDigest
    key_diff: ScratchDiffSummary
    value_diff: ScratchDiffSummary
```

Validate:

- selector is exactly `key` or `value`;
- every shape dimension and byte count is a non-negative integer;
- SHA-256 is exactly 64 lowercase hex characters;
- `equal_to_s0=True` requires zero mismatches, no first mismatch, and zero
  maximum absolute difference;
- `equal_to_s0=False` requires a positive mismatch count and all four
  non-negative location fields;
- checkpoint sequence is exactly S0-S7 per rank, except S3 may repeat once
  per captured segment and must carry `segment_ordinal`;
- every checkpoint must have `synchronized=True`.

Represent repeated S3 rows with an additional optional
`segment_ordinal: int | None` field. S0, S1, S2, S4, S5, S6, and S7 must have
`segment_ordinal=None`.

- [ ] **Step 6: Add failing aggregation and route-classifier tests**

Add fixtures for four ranks and assert:

```python
def test_tp4_aggregation_uses_maximum_not_average():
    rows = make_phase_rows(
        capture_body_by_rank=(100, 110, 120, 900),
        segment_total_by_rank=(200, 210, 220, 1_000),
    )
    aggregate = aggregate_tp4_phase_rows(rows)
    assert aggregate["capture_body_ns"] == 900
    assert aggregate["segment_total_ns"] == 1_000
```

Add complete classifier cases:

- missing rank, source mismatch, verifier disagreement, or cleanup not
  `CLEAN` -> `INCOMPLETE`;
- failed immediate restore round trip -> pivot with
  `scratch_restore_primitive`;
- rank-consistent first divergence plus one exact source location, one
  bounded repair statement, cleanup `CLEAN`, and conservative projected
  timing within both ceilings -> `REPAIR_CANDIDATE`;
- no rank-consistent slow phase, more than one repair, projected maximum
  segment above `1_800_000_000`, projected lifecycle above
  `4_500_000_000`, more than four projected graphs, or isolated-pool memory
  gate failure -> pivot;
- any attempted Phase A1 `GO_SEGMENTED_REPAIR` input -> validation error.

- [ ] **Step 7: Implement deterministic aggregation and classification**

Add:

```python
@dataclass(frozen=True)
class AttributionDiagnosis:
    first_scratch_divergence: str
    slow_capture_phase: str
    root_cause_kind: str
    source_path: str
    source_symbol: str
    repair_statement: str
    repair_count: int
    projected_max_segment_ns: int
    projected_lifecycle_ns: int
    projected_graph_count: int
```

`classify_phase_a1` must return:

```python
{
    "schema_version": "tinyllmforge.tp4-segmented-attribution-decision.v1",
    "phase": "A1",
    "classification": classification,
    "failed_gates": sorted(failed_gates),
    "first_scratch_divergence": first_scratch_divergence,
    "slow_capture_phase": slow_capture_phase,
    "diagnosis_sha256": canonical_sha256(asdict(diagnosis)),
}
```

The classifier must fail closed. `REPAIR_CANDIDATE` requires exactly one
repair, exact source path and symbol, TP-wide rank agreement, exact
S2/S4/S6/S7 checkpoints for any timing evidence used in route selection,
conservative projected compliance with both timing ceilings, at most four
graphs, memory-gate compliance, and `CLEAN` cleanup. A localized scratch
mismatch may support the source diagnosis, but it must not make timing rows
eligible or be silently waived. The classifier must never infer projected
savings by subtracting unmeasured work.

- [ ] **Step 8: Run pure tests GREEN**

Run:

```bash
pytest -q tools/test_segmented_capture_attribution.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit the pure contracts**

Run:

```bash
git add tinyvllm/engine/segmented_capture_attribution.py tools/test_segmented_capture_attribution.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): add capture attribution contracts" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

Expected: one commit containing exactly those two files.

---

### Task 2: Add deterministic scratch-KV forensic primitives

**Files:**
- Create: `tools/tp4_segmented_capture_attribution_worker.py`
- Create: `tools/test_tp4_segmented_capture_attribution_worker.py`
- Read only: `tinyvllm/engine/model_runner.py`
- Read only: `tools/tp4_segmented_capture_census_worker.py`

**Interfaces:**
- Consumes: `ModelRunner.snapshot_kv_slots(physical_slots: list[int]) -> dict[str, torch.Tensor]`
- Consumes: `ModelRunner.restore_kv_slots(physical_slots: list[int], snapshot: dict[str, torch.Tensor]) -> None`
- Produces: `fill_scratch_sentinel(runner, scratch_slots: list[int], *, run_tag: str, rank: int, torch_module) -> None`
- Produces: `snapshot_scratch_checkpoint(runner, scratch_slots: list[int], *, checkpoint: str, rank: int, s0: dict | None, synchronized: bool, segment_ordinal: int | None, torch_module) -> dict`
- Produces: `first_scratch_divergence(rows: list[dict]) -> str | None`
- Produces: `_AttributionCudaBackend`

- [ ] **Step 1: Write failing deterministic-sentinel tests**

Create a fake runner whose key/value cache tensors have deliberately different
shapes and verify:

```python
def test_sentinel_is_nonzero_deterministic_and_rank_sensitive():
    first = make_fake_runner()
    second = make_fake_runner()
    other_rank = make_fake_runner()
    fill_scratch_sentinel(
        first, SLOTS, run_tag="a1-r1", rank=0, torch_module=torch
    )
    fill_scratch_sentinel(
        second, SLOTS, run_tag="a1-r1", rank=0, torch_module=torch
    )
    fill_scratch_sentinel(
        other_rank, SLOTS, run_tag="a1-r1", rank=1, torch_module=torch
    )
    assert snapshots_equal(first, second)
    assert not snapshots_equal(first, other_rank)
    assert every_tensor_has_nonzero_value(first)
```

The fake runner must expose the same `snapshot_kv_slots` and
`restore_kv_slots` calls as `ModelRunner`; tests must not depend on a GPU.

- [ ] **Step 2: Run the sentinel test and verify RED**

Run:

```bash
pytest -q tools/test_tp4_segmented_capture_attribution_worker.py \
  -k sentinel
```

Expected: import or symbol failure for the new worker.

- [ ] **Step 3: Implement deterministic sentinel filling**

Use a counter-based integer construction rather than Python's randomized
`hash()`. Derive a 64-bit seed with SHA-256 over:

```text
run_tag | rank | key-or-value | layer | scratch-slot-ordinal
```

Then combine flattened element offset with an odd 64-bit multiplier, map into
a finite nonzero range, cast to the destination dtype, and copy only the
selected physical slots. Preserve each tensor's device, shape, and dtype.
Synchronize before returning.

Do not serialize the generated tensor contents.

- [ ] **Step 4: Write failing digest and bounded-diff tests**

Test exact and mismatching snapshots for both keys and values. Assert:

- digest includes canonical CPU bytes, dtype, shape, and byte count;
- changing one value changes the SHA-256;
- first mismatch maps back to layer, scratch-slot ordinal, head, and remaining
  flattened element offset;
- only the first mismatch and aggregate count/max difference are emitted;
- no `bytes`, `data`, `tensor`, or base64 payload field exists.

- [ ] **Step 5: Implement checkpoint snapshot and diff helpers**

Before every snapshot, call `torch.cuda.synchronize()` in the real backend and
pass `synchronized=True`. Canonical digest order must be:

```text
selector -> layer order -> scratch-slot order -> remaining tensor dimensions
```

Move only the eight selected scratch slots to CPU for hashing. Record CPU
snapshot/hash duration separately as `scratch_snapshot_cpu_ns`; do not include
that value in CUDA capture phase durations, but keep it inside
`program_lifecycle_ns`.

- [ ] **Step 6: Write failing restore-round-trip and S0-S7 ordering tests**

Using a fake backend call log, require:

```text
sentinel -> sync -> S0
restore(S0) -> sync -> round_trip
eager -> sync -> S1
restore(S0) -> sync -> S2
capture segment -> sync -> S3(segment ordinal)
restore(S0) -> sync -> S4
replay -> sync -> S5
restore(S0) -> sync -> S6
reverse reset -> final sync -> S7
```

Assert that:

- an immediate round-trip mismatch stops before eager/capture;
- an eager error remains the raised primary error even if final restore or
  reset also fails;
- every created graph resets once in reverse order;
- reset remains idempotent;
- S7 is attempted after an operational failure;
- non-root ranks may return `logits=None`.

- [ ] **Step 7: Implement the forensic backend skeleton**

Copy no large census implementation wholesale. Import the stable segment plan
and state comparison helpers where practical, and implement a focused
`_AttributionCudaBackend` that:

- allocates the same eight scratch slots as r60;
- initializes the sentinel before S0;
- performs the immediate restore round trip;
- records S0-S7 through one checkpoint method;
- preserves selected and unselected state snapshots;
- keeps source errors primary while attaching restore/reset failures;
- exposes `checkpoint_rows`, `memory_snapshot`, and
  `stable_boundary_buffer_bytes`;
- cannot be installed into normal production dispatch.

- [ ] **Step 8: Run scratch forensic tests GREEN**

Run:

```bash
pytest -q \
  tools/test_segmented_capture_attribution.py \
  tools/test_tp4_segmented_capture_attribution_worker.py \
  -k "scratch or sentinel or checkpoint or restore or reset or non_root"
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit scratch forensics**

Run:

```bash
git add tools/tp4_segmented_capture_attribution_worker.py tools/test_tp4_segmented_capture_attribution_worker.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): add scratch KV forensics" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

Expected: one commit containing exactly the worker skeleton and its tests.

---

### Task 3: Instrument non-overlapping capture phases and metadata

**Files:**
- Modify: `tools/tp4_segmented_capture_attribution_worker.py`
- Modify: `tools/test_tp4_segmented_capture_attribution_worker.py`
- Read only: `tinyvllm/layers/qwen35_packed_layer_stack.py`
- Read only: `tinyvllm/models/qwen35_packed.py`

**Interfaces:**
- Produces: `_CapturedAttributionSegment`
- Produces: `capture_attributed_segment(segment, *, ordinal: int, control_id: str, pool_mode: str, shared_pool: object | None) -> _CapturedAttributionSegment`
- Produces each required timing field:
  - `snapshot_and_prepare_ns`
  - `graph_object_create_ns`
  - `capture_context_enter_ns`
  - `capture_body_ns`
  - `capture_context_exit_and_instantiate_ns`
  - `post_capture_synchronize_ns`
  - `post_capture_restore_ns`
  - `graph_reset_ns`
  - `segment_total_ns`
  - `program_lifecycle_ns`

- [ ] **Step 1: Write a failing timestamp-boundary test**

Use a deterministic incrementing clock and fake context managers. Assert the
exact call order:

```text
segment_started
graph_create_started
graph_create_finished
capture_enter_started
first_inside_capture
last_inside_capture
after_capture_exit
after_cuda_synchronize
after_restore
after_reset
segment_finished
```

Assert that `capture_context_enter_ns` measures only entry overhead,
`capture_body_ns` is first-inside to last-inside,
`capture_context_exit_and_instantiate_ns` is last-inside to after-exit, and
`post_capture_synchronize_ns` is after-exit to after-sync.

- [ ] **Step 2: Run the timing test and verify RED**

Run:

```bash
pytest -q tools/test_tp4_segmented_capture_attribution_worker.py \
  -k phase_timestamps
```

Expected: failure because attributed capture is not implemented.

- [ ] **Step 3: Implement capture phase timing**

Add `_CapturedAttributionSegment` with:

```python
@dataclass(frozen=True)
class _CapturedAttributionSegment:
    graph: object
    pool_identity: str
    accounting: CapturePhaseAccounting
    metadata: dict
```

Create the graph outside the capture context, timestamp immediately before
and after `torch.cuda.CUDAGraph()`, immediately before context entry, as the
first and last host operations inside `torch.cuda.graph`, immediately after
context exit, and immediately after explicit CUDA synchronization.

Do not claim a separate graph-instantiation interval. Use the exact field
`capture_context_exit_and_instantiate_ns`.

- [ ] **Step 4: Write failing metadata inventory tests**

For each half-open range, assert:

```text
[0,16)   -> 12 linear, 4 full
[16,32)  -> 12 linear, 4 full
[32,48)  -> 12 linear, 4 full
[48,64)  -> 12 linear, 4 full
```

Require every row to include:

- segment ordinal and range;
- candidate convolution/recurrent tensor count and bytes;
- stable hidden/candidate/logits bytes;
- allocated/reserved before, after, and deltas;
- capture-pool identity and `shared`/`isolated` mode;
- current CUDA stream identity;
- collective counts by available receipt operation class, or an explicit
  `unavailable_reason`;
- source SHA and plan SHA.

- [ ] **Step 5: Implement exact metadata collection**

Compute layer types from the model's existing layer metadata or modules,
rather than hard-coding the expected 12:4 result into production diagnostic
code. Count tensors by identity so aliased buffers are not double-counted.

Pool identity must be a run-local opaque digest, never `id(pool)` alone in an
artifact. Hash:

```text
run_tag | rank | pool_mode | pool_ordinal
```

If collective receipts are unavailable, emit:

```python
{
    "available": False,
    "counts": {},
    "unavailable_reason": "existing_receipt_not_exposed",
}
```

Do not fabricate zero collectives.

- [ ] **Step 6: Write failing accounting-integrity tests**

Mutate each phase to a negative value, drop each required field, make
`segment_total_ns` smaller than measured capture, and make lifecycle smaller
than any segment total. Assert fail-closed errors.

Also assert the diagnostics keep scratch hashing and eager-prefix preparation
inside lifecycle while exposing them as separate overhead fields.

- [ ] **Step 7: Run focused worker tests GREEN**

Run:

```bash
pytest -q \
  tools/test_segmented_capture_attribution.py \
  tools/test_tp4_segmented_capture_attribution_worker.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit phase instrumentation**

Run:

```bash
git add tools/tp4_segmented_capture_attribution_worker.py tools/test_tp4_segmented_capture_attribution_worker.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): attribute segmented capture phases" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 4: Implement the bounded stitched, isolated-range, and pool matrix

**Files:**
- Modify: `tools/tp4_segmented_capture_attribution_worker.py`
- Modify: `tools/test_tp4_segmented_capture_attribution_worker.py`

**Interfaces:**
- Produces: `build_phase_a1_controls() -> tuple[dict, ...]`
- Produces: `run_stitched_repeat(backend, *, repeat_ordinal: int) -> dict`
- Produces: `run_isolated_range(backend, *, start_layer: int, end_layer: int, pool_mode: str) -> dict`
- Produces: `select_pool_control_ranges(isolated_rows: list[dict]) -> tuple[tuple[int, int], tuple[int, int]]`
- Produces: `run_phase_a1_matrix(backend) -> dict`

- [ ] **Step 1: Write failing matrix-bound tests**

Assert the fixed first-stage matrix contains:

```python
(
    ("stitched_p4_repeat_0", ((0, 16), (16, 32), (32, 48), (48, 64))),
    ("stitched_p4_repeat_1", ((0, 16), (16, 32), (32, 48), (48, 64))),
    ("isolated_0_16", ((0, 16),)),
    ("isolated_16_32", ((16, 32),)),
    ("isolated_32_48", ((32, 48),)),
    ("isolated_48_64", ((48, 64),)),
)
```

Then assert exactly four pool-control captures are appended: fastest/shared,
fastest/isolated, slowest/shared, and slowest/isolated. Ties resolve by
lexicographically smallest `(start_layer, end_layer)`. No partition search,
8-way plan, 16-way plan, or autotuning entry is allowed.

- [ ] **Step 2: Run matrix tests and verify RED**

Run:

```bash
pytest -q tools/test_tp4_segmented_capture_attribution_worker.py \
  -k "matrix or pool_control or isolated_range"
```

Expected: missing-symbol failures.

- [ ] **Step 3: Implement the fixed matrix and deterministic selection**

Use TP-wide isolated `segment_total_ns` maxima to select fastest and slowest
ranges. Because all ranks must choose the same follow-up controls, gather and
validate the four isolated rows before starting pool controls. Rank
disagreement is an operational error and yields an incomplete worker bundle.

- [ ] **Step 4: Write failing isolated-prefix tests**

For isolated `[start,end)`:

- restore S0 and selected model state;
- eagerly execute `[0,start)` to create the exact hidden input;
- synchronize;
- record `eager_prefix_prepare_ns`;
- start the isolated measured segment lifecycle only after prefix preparation;
- capture exactly `[start,end)`;
- never report eager-prefix time as capture time;
- retain eager-prefix time in total diagnostic cost.

For `[0,16)`, assert prefix duration is present and zero.

- [ ] **Step 5: Implement isolated input preparation**

Use existing range-bounded Qwen3.8 hooks:

```python
hidden = model.embed_exact_graph_inputs(static_input_ids)
model.run_exact_cuda_graph_layer_range(
    state_slot_ids=state_slot_ids,
    token_counts=token_counts,
    position_ids=static_positions,
    hidden_states=hidden,
    start_layer=0,
    end_layer=start_layer,
)
```

Do not commit prefix candidates, mutate the production graph cache, or treat
the isolated result as a complete model output.

- [ ] **Step 6: Write failing stitched-repeat boundary tests**

Require an explicit:

```text
restore S0 -> reset prior graphs -> synchronize -> begin repeat 1
```

boundary. Repeat 1 is diagnostic and must be labeled
`formal_route_row=False`; repeat 0 remains the formal stitched row.

- [ ] **Step 7: Implement pool controls and memory cost**

For the fastest and slowest isolated ranges:

- shared mode reuses the bounded diagnostic shared pool;
- isolated mode passes no shared pool and records the graph's resolved pool;
- reset every graph after its row is finalized;
- record allocated/reserved deltas and stable buffer bytes;
- expose `isolated_pool_memory_gate_pass` against the existing frozen
  production memory limit already used by the census verifier.

- [ ] **Step 8: Run matrix tests GREEN**

Run:

```bash
pytest -q tools/test_tp4_segmented_capture_attribution_worker.py
```

Expected: all worker tests pass.

- [ ] **Step 9: Commit the bounded matrix**

Run:

```bash
git add tools/tp4_segmented_capture_attribution_worker.py tools/test_tp4_segmented_capture_attribution_worker.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): add bounded attribution matrix" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 5: Add the Phase A1 worker artifact and process lifecycle

**Files:**
- Modify: `tools/tp4_segmented_capture_attribution_worker.py`
- Modify: `tools/test_tp4_segmented_capture_attribution_worker.py`
- Read only: `tools/tp4_segmented_capture_census_worker.py`

**Interfaces:**
- Produces: `build_engine_config() -> dict`
- Produces: `run_phase_a1(*, model_root, run_tag: str, timeout_s: float, engine_factory=None, workload_runner=None) -> dict`
- Produces CLI:
  - `--model-root`
  - `--run-tag`
  - `--output-root`
  - `--timeout-s`
- Produces worker files:
  - `phase_rows.jsonl`
  - `scratch_rows.jsonl`
  - `rank_results.json`
  - `process_receipts.json`
  - `worker_summary.json`

- [ ] **Step 1: Write failing worker orchestration tests**

Adapt the census worker's fake-engine pattern and assert:

- Qwen3.8 frozen config, TP4, batch 8, max tokens 2, model length 384;
- manual diagnostic capture only;
- each rank is armed and acknowledged;
- one decode workload triggers Phase A1 once;
- four-rank results are collected;
- engine exit occurs after success and failure;
- rendezvous retry occurs only for address-in-use errors;
- active children are reaped only when owned by the worker;
- no worker output path escapes its supplied output root.

- [ ] **Step 2: Run orchestration tests and verify RED**

Run:

```bash
pytest -q tools/test_tp4_segmented_capture_attribution_worker.py \
  -k "orchestration or engine or config or rendezvous or output"
```

Expected: failures for missing orchestration.

- [ ] **Step 3: Implement one-shot worker orchestration**

Follow the existing census engine construction and cleanup pattern, but call
only `run_phase_a1_matrix`. Ensure:

- one fresh engine for the entire bounded matrix so lazy-init and shared-pool
  effects remain observable;
- no second process launch on timeout or disconnect;
- process receipt records rank, PID, process-group identity, exit code,
  destruction acknowledgement, and first error;
- atomic JSON/JSONL writes use temporary files below `output_root`;
- JSON uses `sort_keys=True` and `allow_nan=False`.

- [ ] **Step 4: Write failing rank-row and worker-summary tests**

Require exactly ranks 0-3, identical source/plan/control identity, all fixed
controls, valid S0-S7 sequences per control, and no duplicate row IDs.

Worker summary must include:

```python
{
    "schema_version": "tinyllmforge.tp4-segmented-attribution-worker.v1",
    "phase": "A1",
    "run_tag": "phase-a1-test",
    "complete": True,
    "benefit": {
        "attributed_segments": 16,
        "first_scratch_divergence": "S4",
        "restore_round_trip_exact": True,
    },
    "cost": {
        "diagnostic_capture_count": 16,
        "diagnostic_synchronization_count": 32,
        "total_worker_duration_ns": 1_000,
        "scratch_snapshot_cpu_ns": 100,
        "peak_allocated_delta_bytes": 200,
        "peak_reserved_delta_bytes": 300,
    },
}
```

- [ ] **Step 5: Implement atomic artifact writing and CLI**

Write only bounded summaries and hashes. Never write full scratch tensors.
On failure, persist all already-completed rows, the first operational error,
cleanup results, and `complete=False`, then return a nonzero CLI exit code.

- [ ] **Step 6: Run worker tests GREEN**

Run:

```bash
pytest -q \
  tools/test_segmented_capture_attribution.py \
  tools/test_tp4_segmented_capture_attribution_worker.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit worker orchestration**

Run:

```bash
git add tools/tp4_segmented_capture_attribution_worker.py tools/test_tp4_segmented_capture_attribution_worker.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): emit attribution worker evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 6: Add the independent Phase A1 verifier

**Files:**
- Create: `tools/verify_tp4_segmented_capture_attribution.py`
- Create: `tools/test_verify_tp4_segmented_capture_attribution.py`
- Read only: `tools/verify_tp4_segmented_capture_census.py`

**Interfaces:**
- Produces: `verify_bundle(bundle_or_root) -> dict`
- Produces CLI: `--bundle-root`
- Consumes all worker rows, source identity, workload, admission, process
  receipts, cleanup receipts, manifest, and diagnosis.

- [ ] **Step 1: Write a complete valid-bundle fixture**

Create a small synthetic four-rank bundle containing all fixed controls,
phase rows, S0-S7 scratch rows, source identity, admission, process receipts,
cleanup, diagnosis, and manifest entries. Keep durations tiny but preserve
all required accounting relations.

- [ ] **Step 2: Write failing verification tests**

Cover:

- exact frozen model/revision/workload;
- phase `A1`;
- committed 40-hex source revision and 64-hex source-tree digest;
- four strict-clean GPUs with empty compute-process lists;
- ranks exactly 0-3;
- required controls and no extras;
- non-negative phase accounting and TP-wide maxima;
- exact checkpoint order, synchronization, digest, and bounded diff fields;
- fastest/slowest control selection reconstructed from isolated TP maxima;
- repeat 1 excluded from formal timing eligibility;
- graph reset and process-group destruction;
- at least three final empty exact-tag scans;
- manifest hash mismatch;
- modified phase or scratch row after manifest creation;
- `GO_SEGMENTED_REPAIR` rejected in Phase A1;
- local/remote verifier result equality.

- [ ] **Step 3: Run verifier tests and verify RED**

Run:

```bash
pytest -q tools/test_verify_tp4_segmented_capture_attribution.py
```

Expected: import failure for the new verifier.

- [ ] **Step 4: Implement bundle loading and structural verification**

Use a separate schema:

```text
tinyllmforge.tp4-segmented-attribution-bundle.v1
```

Require these files:

```text
source_identity.json
plan.json
admission.json
phase_rows.jsonl
scratch_rows.jsonl
rank_results.json
process_receipts.json
cleanup_receipt.json
diagnosis.json
worker_summary.json
manifest.json
```

Do not import producer-side classification output. Import only the
dependency-light pure contract and independently reconstruct the decision
from raw rows.

- [ ] **Step 5: Implement phase, scratch, matrix, and rank reconstruction**

For every segment/control:

- reconstruct phase accounting;
- aggregate with maximum rank duration, never average;
- require identical range/type/candidate/pool/source metadata across ranks
  where applicable;
- reconstruct first scratch divergence;
- verify immediate restore round trip;
- reconstruct fastest and slowest isolated controls;
- calculate shared-versus-isolated pool deltas;
- calculate exact benefit and cost summaries.

If any required evidence is missing or disagrees, return `INCOMPLETE` with a
specific failed gate.

- [ ] **Step 6: Implement independent Phase A1 classification**

Call `classify_phase_a1` only with verifier-reconstructed evidence. Emit:

```python
{
    "schema_version": "tinyllmforge.tp4-segmented-attribution-verification.v1",
    "phase": "A1",
    "classification": classification,
    "failed_gates": sorted(failed_gates),
    "source_revision": source_revision,
    "run_tag": run_tag,
    "tp_wide_phase_summary": tp_wide_phase_summary,
    "scratch_summary": scratch_summary,
    "pool_summary": pool_summary,
    "benefit": benefit,
    "cost": cost,
}
```

Ensure classifier reasons name the exact failing boundary or ceiling.

- [ ] **Step 7: Run verifier tests GREEN**

Run:

```bash
pytest -q \
  tools/test_segmented_capture_attribution.py \
  tools/test_verify_tp4_segmented_capture_attribution.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit the verifier**

Run:

```bash
git add tools/verify_tp4_segmented_capture_attribution.py tools/test_verify_tp4_segmented_capture_attribution.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): verify capture attribution evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 7: Add the strict-clean Phase A1 controller and manifests

**Files:**
- Create: `tools/run_tp4_segmented_capture_attribution.py`
- Create: `tools/test_run_tp4_segmented_capture_attribution.py`
- Read only: `tools/run_tp4_segmented_capture_census.py`
- Read only: `tools/run_tp4_decode_replay.py`

**Interfaces:**
- Produces: `build_plan(*, run_tag: str, source_identity: dict, selected_gpus: list[dict], admission_mode: str) -> dict`
- Produces: `monitor_and_run(seed: dict, adapter: object) -> dict`
- Produces: `ProductionAdapter`
- Produces pre-verification `manifest.json`
- Produces post-verification `post_verification_manifest.json`
- Produces CLI flags matching the established census controller where
  applicable.

- [ ] **Step 1: Write failing plan/path/source tests**

Assert:

- run tag matches `^[A-Za-z0-9][A-Za-z0-9._-]*$` and excludes `..`;
- local attempt root and remote attempt root must not already exist;
- every remote path and environment variable is beneath:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
```

- source revision is committed and source tree hash is bound;
- model repository/revision and Phase A1 schema are exact;
- admission mode must be `strict_clean`;
- exactly four unique GPU indices/UUIDs are admitted;
- memory, utilization, and compute-process checks retain existing thresholds;
- no path defaults to `/tmp`, `$HOME`, or `/`.

- [ ] **Step 2: Run controller tests and verify RED**

Run:

```bash
pytest -q tools/test_run_tp4_segmented_capture_attribution.py
```

Expected: import failure for the new controller.

- [ ] **Step 3: Implement plan construction and source freezing**

Reuse established helpers where they preserve identical semantics. The
controller's remote root is:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/tp4-segmented-capture-attribution
```

Bind the exact committed source revision, tree hash, worker/verifier hashes,
model identity, workload, fixed controls, timing limits, memory limit, and
Phase A1 classifier schema into `plan.json`.

- [ ] **Step 4: Write failing pipeline and retry tests**

Use a fake adapter to assert exact ordering:

```text
freeze_source
ssh_storage_preflight
kerberos_ttl_guard
gpu_admission
launch_once
wait
owned_cleanup
download
remote_verify
local_verify
validate_verifier_identity
write_post_verification_manifest
final_live_exact_tag_scan
```

Assert:

- cleanup runs after every launch path;
- SSH 255 retries only idempotent source/archive operations;
- a worker disconnect never calls launch again;
- insufficient Kerberos TTL exits before GPU admission or launch;
- foreign processes are reported but untouched;
- only exact-tag-owned PIDs can be reaped;
- first operational error remains primary when cleanup also fails.

- [ ] **Step 5: Implement production adapter and one-launch lifecycle**

Reuse the census controller's:

- SSH argument builder;
- local Kerberos query and guard margin;
- mounted-storage preflight;
- strict-clean polling;
- source archive staging;
- process-group ownership receipts;
- exact-tag scanner/reaper;
- bounded archive retry.

Do not subclass behavior that hard-codes the census schemas or filenames.
Extract no shared helper in this task unless a RED test proves the duplicate
implementation would diverge; preserving the frozen census source is more
important than deduplication.

- [ ] **Step 6: Write failing manifest and verifier-agreement tests**

Require `manifest.json` to bind every producer input and artifact except the
two verifier outputs. Require `post_verification_manifest.json` to bind:

- pre-verification manifest digest;
- remote verifier bytes and SHA-256;
- local verifier bytes and SHA-256;
- byte-identical verifier result;
- final cleanup receipt;
- final live exact-tag scan.

Mutating one phase row, scratch row, process receipt, diagnosis, or verifier
output must fail.

- [ ] **Step 7: Implement manifests and terminal result**

The controller returns success only when:

- worker artifacts are structurally complete;
- cleanup is `CLEAN`;
- remote and local verifier bytes are identical;
- both classify the same terminal state;
- post-verification manifest validates;
- final live exact-tag scan is empty.

`REPAIR_CANDIDATE` and
`PIVOT_TP4_COMMUNICATION_COMPUTE_FUSION` are successful terminal diagnostic
outcomes. `INCOMPLETE` is persisted but the controller exits nonzero.

- [ ] **Step 8: Run controller tests GREEN**

Run:

```bash
pytest -q \
  tools/test_run_tp4_segmented_capture_attribution.py \
  tools/test_verify_tp4_segmented_capture_attribution.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit the controller**

Run:

```bash
git add tools/run_tp4_segmented_capture_attribution.py tools/test_run_tp4_segmented_capture_attribution.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): orchestrate strict capture attribution" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 8: Run local validation and source review

**Files:**
- Verify only:
  - `tinyvllm/engine/segmented_capture_attribution.py`
  - `tools/tp4_segmented_capture_attribution_worker.py`
  - `tools/run_tp4_segmented_capture_attribution.py`
  - `tools/verify_tp4_segmented_capture_attribution.py`
  - their four focused test files

- [ ] **Step 1: Run all new focused tests**

Run:

```bash
pytest -q \
  tools/test_segmented_capture_attribution.py \
  tools/test_tp4_segmented_capture_attribution_worker.py \
  tools/test_run_tp4_segmented_capture_attribution.py \
  tools/test_verify_tp4_segmented_capture_attribution.py
```

Expected: all tests pass.

- [ ] **Step 2: Run adjacent frozen census tests**

Run:

```bash
pytest -q \
  tools/test_segmented_exact_cuda_graph.py \
  tools/test_tp4_segmented_capture_census_worker.py \
  tools/test_run_tp4_segmented_capture_census.py \
  tools/test_verify_tp4_segmented_capture_census.py
```

Expected: the existing suite remains green; no r60 contract changes.

- [ ] **Step 3: Compile the new source without polluting the repository**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-phase-a1-pycache \
python -m py_compile \
  tinyvllm/engine/segmented_capture_attribution.py \
  tools/tp4_segmented_capture_attribution_worker.py \
  tools/run_tp4_segmented_capture_attribution.py \
  tools/verify_tp4_segmented_capture_attribution.py
```

Expected: exit code 0.

- [ ] **Step 4: Run focused static self-review**

Run:

```bash
rg -n \
  'kinit|krenew|git add -A|/tmp/|/root|kill -9|pkill|GO_SEGMENTED_REPAIR' \
  tinyvllm/engine/segmented_capture_attribution.py \
  tools/tp4_segmented_capture_attribution_worker.py \
  tools/run_tp4_segmented_capture_attribution.py \
  tools/verify_tp4_segmented_capture_attribution.py
```

Expected:

- no credential-refresh command;
- no broad staging command;
- no unsafe remote temporary path;
- no foreign-process kill path;
- any `GO_SEGMENTED_REPAIR` occurrence is only a rejection test or explicit
  Phase A1 invalid-state guard.

- [ ] **Step 5: Review the exact source diff**

Run:

```bash
git diff --check HEAD --
git diff --stat HEAD -- \
  tinyvllm/engine/segmented_capture_attribution.py \
  tools/tp4_segmented_capture_attribution_worker.py \
  tools/run_tp4_segmented_capture_attribution.py \
  tools/verify_tp4_segmented_capture_attribution.py \
  tools/test_segmented_capture_attribution.py \
  tools/test_tp4_segmented_capture_attribution_worker.py \
  tools/test_run_tp4_segmented_capture_attribution.py \
  tools/test_verify_tp4_segmented_capture_attribution.py
```

Inspect every remaining unstaged change in those exact paths. Do not inspect
or normalize unrelated artifacts.

- [ ] **Step 6: Run code review**

Use the repository's code-review workflow on only the Phase A1 source and
tests. Resolve every high-confidence correctness, lifecycle, source-binding,
cleanup, or evidence-integrity finding through a new RED/GREEN cycle.

- [ ] **Step 7: Commit review fixes if needed**

Stage only changed Phase A1 paths and commit:

```bash
git -c core.hooksPath=/dev/null commit \
  -m "fix(tp4): harden capture attribution gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

Skip this commit if review requires no source changes.

---

### Task 9: Push and verify the exact diagnostic source

**Files:**
- No file changes expected.

- [ ] **Step 1: Confirm exact-path cleanliness**

Run:

```bash
git status --short -- \
  tinyvllm/engine/segmented_capture_attribution.py \
  tools/tp4_segmented_capture_attribution_worker.py \
  tools/run_tp4_segmented_capture_attribution.py \
  tools/verify_tp4_segmented_capture_attribution.py \
  tools/test_segmented_capture_attribution.py \
  tools/test_tp4_segmented_capture_attribution_worker.py \
  tools/test_run_tp4_segmented_capture_attribution.py \
  tools/test_verify_tp4_segmented_capture_attribution.py
```

Expected: no output.

- [ ] **Step 2: Push only the approved branch**

Run:

```bash
git push origin feat/kv-sparse-attention
```

Expected: push succeeds.

- [ ] **Step 3: Verify local, tracking, and remote SHA equality**

Run:

```bash
git rev-parse HEAD
git rev-parse origin/feat/kv-sparse-attention
git ls-remote origin refs/heads/feat/kv-sparse-attention
```

Expected: all three SHA values are identical. Record this SHA as the only
eligible Phase A1 source revision.

---

### Task 10: Execute one fresh-tag strict-clean Phase A1 run

**Files:**
- Create locally only under:
  - `experiments/qwen35_hybrid_state/<fresh-phase-a1-tag>/`
- Create remotely only under:
  - `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/tp4-segmented-capture-attribution/<fresh-phase-a1-tag>/`

**Interfaces:**
- Consumes the pushed source SHA from Task 9.
- Produces one immutable Phase A1 bundle and no production runtime change.

- [ ] **Step 1: Choose and prove a fresh tag**

Use the next unused monotonically increasing tag, for example:

```text
20260907-qwen38-tp4-segmented-capture-attribution-r61
```

Before launch, prove the tag is absent locally and remotely. If it exists in
either place, increment the suffix; never reuse or overwrite it.

- [ ] **Step 2: Run preflight without refreshing credentials**

Run the controller's dry/preflight path. Require:

- sufficient Kerberos TTL for the full bounded run plus the existing guard;
- mounted remote base;
- exact committed source;
- model snapshot at the frozen revision;
- sufficient local and remote artifact space;
- four `strict_clean` GPUs;
- no exact-tag-owned stale process.

If any check fails, persist a prelaunch `INCOMPLETE` receipt and launch
nothing.

- [ ] **Step 3: Launch exactly once**

Run:

```bash
python tools/run_tp4_segmented_capture_attribution.py \
  --run-tag <fresh-phase-a1-tag> \
  --admission-mode strict_clean
```

Expected: one remote process group is created for the exact tag. Do not start
a second launch after SSH interruption; inspect the existing supervisor,
process receipt, and artifact directory.

- [ ] **Step 4: Monitor state transitions**

Inspect only meaningful transitions:

```text
PRELAUNCH -> ADMITTED -> RUNNING -> WORKER_TERMINAL
-> CLEANUP_TERMINAL -> DOWNLOADED -> VERIFIED -> TERMINAL
```

Do not treat silence, elapsed time, or an open SSH ControlMaster as
scientific progress. Do not terminate foreign work.

- [ ] **Step 5: Require exact-tag cleanup**

After worker termination:

- all ranks exit;
- process groups acknowledge destruction;
- no owned child remains;
- at least three archived exact-tag scans are empty;
- one final live exact-tag scan is empty;
- cleanup classification is `CLEAN`.

If cleanup is not clean, the run is `INCOMPLETE` regardless of measurements.

- [ ] **Step 6: Run remote and local independent verification**

The controller must run the committed verifier remotely and the local
verifier against downloaded bounded artifacts. Require byte-identical JSON
and matching SHA-256 values.

- [ ] **Step 7: Validate the post-verification manifest**

Require every producer artifact, both verifier outputs, cleanup receipt, and
final live scan to match the post-verification manifest. Any mismatch yields
`INCOMPLETE`.

- [ ] **Step 8: Record the terminal classifier**

Accept exactly one:

```text
REPAIR_CANDIDATE
PIVOT_TP4_COMMUNICATION_COMPUTE_FUSION
INCOMPLETE
```

Do not continue into repair implementation during this task.

---

### Task 11: Publish the Phase A1 audit and handoff

**Files:**
- Modify: `docs/superpowers/audits/2026-08-31-tp4-collective-stable-decode-replay-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [ ] **Step 1: Write the evidence table**

Add:

- source commit and source-tree digest;
- immutable run tag;
- admission inventory;
- control/rank inventory;
- TP-wide phase maxima for every segment/control;
- stitched repeat comparison;
- isolated-range comparison;
- shared-versus-isolated pool comparison;
- first scratch divergence per rank;
- S2/S4/S6/S7 exactness;
- immediate restore-round-trip result;
- exact output/state/reset results;
- allocated/reserved/stable-buffer costs;
- worker and full lifecycle duration;
- verifier hashes and byte equality;
- manifest hashes;
- cleanup receipts and live scan;
- exact terminal classifier and failed gates.

- [ ] **Step 2: State benefit and cost explicitly**

Use this claim boundary verbatim:

```text
capture attribution is not steady-state performance
scratch repair is not replay qualification
a GO diagnosis is not production GO
a pivot is a technically complete negative result
```

Do not claim TTFT, TPOT, P99, throughput, production memory, or end-to-end
benefit from Phase A1.

- [ ] **Step 3: Write the route-specific handoff**

For `REPAIR_CANDIDATE`, record the one source location/symbol and bounded
repair statement, then set the immediate next action to writing a separate
Phase A2 repair design and plan.

For `PIVOT_TP4_COMMUNICATION_COMPUTE_FUSION`, close segmented capture and set
the immediate next action to a separate steady-state communication profile
and fusion design.

For `INCOMPLETE`, record the exact evidence/infrastructure defect and the
minimum correction required before a fresh-tag rerun. Do not select a
technical route.

- [ ] **Step 4: Verify docs and stage exact paths**

Run:

```bash
git diff --check -- \
  docs/superpowers/audits/2026-08-31-tp4-collective-stable-decode-replay-audit.md \
  AGENT_HANDOFF_STATE.md
git add \
  docs/superpowers/audits/2026-08-31-tp4-collective-stable-decode-replay-audit.md \
  AGENT_HANDOFF_STATE.md
git diff --cached --check
```

- [ ] **Step 5: Commit the immutable Phase A1 result**

Run:

```bash
git -c core.hooksPath=/dev/null commit \
  -m "docs(tp4): record capture attribution result" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

- [ ] **Step 6: Verify final SHA equality**

Run:

```bash
git rev-parse HEAD
git rev-parse origin/feat/kv-sparse-attention
git ls-remote origin refs/heads/feat/kv-sparse-attention
```

Expected: local, tracking, and remote SHA values are identical.

## Completion Criteria

Phase A1 is complete only when:

1. all new and adjacent local tests pass;
2. source review has no unresolved high-confidence issue;
3. exact diagnostic source is committed and pushed;
4. one fresh strict-clean Phase A1 tag reaches a terminal worker state;
5. exact-tag cleanup is `CLEAN`;
6. local and remote verifier outputs are byte-identical;
7. pre- and post-verification manifests validate;
8. the audit and handoff are committed and pushed;
9. the terminal state is exactly `REPAIR_CANDIDATE`,
   `PIVOT_TP4_COMMUNICATION_COMPUTE_FUSION`, or `INCOMPLETE`;
10. no Phase A2 repair, production integration, or performance claim has
    started under this plan.
