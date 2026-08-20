# Autoregressive Draft Bounded Rollback Journal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. This workspace explicitly forbids
> subagents and new worktrees.

**Goal:** Replace the two decode-critical-path full-capacity rollback snapshots
with bounded mutation journals while preserving exact Proposal-KV, scheduler,
hybrid-state, publication, and runtime-poisoning behavior.

**Architecture:** `block_manager.py` owns a Proposal-KV batch journal whose
size is proportional to plans, reserved blocks, and publication hashes.
`scheduler.py` owns a prepared-postprocess journal whose size is proportional
to scheduled sequences, their blocks, and Proposal-KV blocks that can be
published immediately before scheduler commit. The engine extends the
scheduler journal with already-prepared KV plans before publication, then
retains the current Proposal-KV → scheduler ordering and poisons runtime if
either journal cannot restore atomically.

**Tech Stack:** Python dataclasses, `deque`, existing TinyLLMForge
`BlockManager`/`Scheduler`/`Sequence`/hybrid-state types, pytest, existing
source-bound command-timeline and paired-gate tooling.

## Global Constraints

- The only authoritative checkout is
  `/Users/bytedance/Desktop/TinyLLMForge`, which resolves to
  `/Users/bytedance/dev/TinyLLMForge`.
- Never modify `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not create another worktree; baseline packaging uses the exact pinned
  source archive or preserved source bundle.
- Push only to `origin/feat/kv-sparse-attention`.
- Never use `git add -A`; stage only the exact files named by the current
  commit step.
- Commit with `git -c core.hooksPath=/dev/null commit`.
- Every commit contains exactly one trailer:
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>`.
- Preserve `PreparedSchedulerPostprocess.snapshot` as a public attribute; its
  value becomes `SchedulerPostprocessJournal`.
- Preserve Proposal-KV validation, ownership, exactly-once state, all-or-none
  batch commit, free-list order, duplicate hash buckets, and exception
  propagation.
- Preserve scheduler rollback for decode, completion, prefill, mixed batches,
  hybrid-state release, hook failure, SLO/progress state, and adaptive
  controller state.
- Do not change model execution, sampling, accepted-token selection, output
  tokens, timeline schema, or campaign identity.
- Do not add CUDA synchronization, `.item()`, worker acknowledgement, fences,
  measured-path logging, profiling, or production GC control.
- No journal may iterate all configured KV blocks or copy the complete
  free/used inventories, hash dictionaries, or hybrid allocator.
- Queue snapshots may copy object references because they are bounded by
  admitted active sequences.
- Rollback failure is terminal: retain the original commit failure as causal
  context, mark the journal rollback-failed, poison speculative runtime, and
  reject journal reuse.
- All remote outputs, caches, logs, diagnostics, receipts, manifests, and
  scratch stay below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Do not write task output to local or remote `/`, `/tmp`, or `/private/tmp`.
- Do not modify `/data00/home/sitian/tllm/TinyLLMForge`.
- Do not refresh Kerberos manually or interfere with unrelated processes.
- A real campaign requires exactly four GPUs, each with memory used
  `<=1024 MiB`, utilization `<=5%`, and no compute processes.
- Failed or attempted run tags are immutable; every attempt uses a fresh tag.
- No performance claim is valid until the same interleaved paired campaign
  passes correctness, stationarity, TPOT, TTFT, and throughput gates.

---

### Task 1: Establish Structural and Proposal-KV RED Tests

**Files:**
- Modify: `tools/test_speculative_kv_transaction.py:58-98`
- Modify: `tools/test_speculative_kv_transaction.py:684-808`

**Interfaces:**
- Consumes: existing `BlockManager.commit_speculative_kv_commit_batch()`.
- Produces: `_IndexableNonIterableBlocks`, a touched-state snapshot helper,
  and deterministic tests that fail while the batch commit traverses all
  blocks or restores from full allocator copies.

- [ ] **Step 1: Add an indexable, non-iterable block collection**

Add this helper after `_restore_sequence_block_size`:

```python
class _IndexableNonIterableBlocks:
    def __init__(self, values):
        self._values = list(values)
        self.index_reads = []

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise AssertionError("block slices are not allowed")
        self.index_reads.append(index)
        return self._values[index]

    def __iter__(self):
        raise AssertionError("full block iteration is not allowed")
```

Add a helper that swaps the manager collection only after setup:

```python
def _guard_block_iteration(manager):
    guarded = _IndexableNonIterableBlocks(manager.blocks)
    manager.blocks = guarded
    return guarded
```

- [ ] **Step 2: Add a success-path complexity test**

Add a two-sequence batch test with `num_blocks=4096`, prepare both plans
before installing the guard, commit, and assert:

```python
guarded = _guard_block_iteration(manager)
manager.commit_speculative_kv_commit_batch(plans)

touched = {
    block_id
    for plan in plans
    for block_id in (
        plan.committed_block_ids
        + plan.unused_block_ids
        + tuple(
            publication.block_id
            for publication in plan.publications
        )
    )
}
assert set(guarded.index_reads).issubset(touched)
assert len(set(guarded.index_reads)) <= len(touched)
```

The assertion permits repeated indexed access during validation but forbids
reads from unrelated capacity.

- [ ] **Step 3: Add failure-atomic touched-state tests**

Extend the existing second-plan failure case into three parametrized
injection points:

```python
@pytest.mark.parametrize(
    "failure_point",
    ("after_publication", "after_unused_release", "second_plan"),
)
def test_commit_speculative_kv_batch_journal_restores_touched_state(
    monkeypatch,
    failure_point,
):
    ...
```

Build duplicate-hash state before the transaction by registering two idle
blocks under the same hash, preserve `_allocator_snapshot(manager)`, then
inject:

- after `_register_cached_block()` has changed primary and duplicate buckets;
- after `release_reserved_blocks()` has appended unused blocks;
- before applying the second plan.

For every injection assert exact equality of:

```python
assert _allocator_snapshot(manager) == allocator_before
assert tuple(seq.block_table for seq in sequences) == tables_before
assert tuple(tx.state for tx in transactions) == (
    "materialized",
    "materialized",
)
```

Also install `_IndexableNonIterableBlocks` after setup so rollback cannot hide
a full traversal.

- [ ] **Step 4: Run the Proposal-KV RED slice**

Run:

```bash
python -m pytest \
  tools/test_speculative_kv_transaction.py \
  -k 'batch and (non_iterable or journal or failure_restores)' \
  -vv
```

Expected: FAIL with `AssertionError: full block iteration is not allowed`
from the existing `for block in self.blocks` snapshot.

- [ ] **Step 5: Preserve RED evidence**

Record the exact command, failing test names, and failure reason in the task
notes used for the later handoff update. Do not edit production code before
the structural failure is observed.

---

### Task 2: Implement the Bounded Proposal-KV Commit Journal

**Files:**
- Modify: `tinyvllm/engine/block_manager.py:57-74`
- Modify: `tinyvllm/engine/block_manager.py:1426-1559`
- Test: `tools/test_speculative_kv_transaction.py`

**Interfaces:**
- Produces:
  - `SpeculativeKVCommitRollbackError(commit_error, rollback_error)`;
  - `_SpeculativeKVBlockState`;
  - `_SpeculativeKVHashState`;
  - `_SpeculativeKVCommitJournal.capture(manager, plans)`;
  - `_SpeculativeKVCommitJournal.rollback(manager)`.
- Consumes: validated `tuple[SpeculativeKVCommitPlan, ...]`.

- [ ] **Step 1: Add local journal value types**

Add below `SpeculativeKVCommitPlan`:

```python
class SpeculativeKVCommitRollbackError(RuntimeError):
    def __init__(self, commit_error, rollback_error):
        super().__init__(
            "speculative KV commit rollback failed: "
            f"{rollback_error}"
        )
        self.commit_error = commit_error
        self.rollback_error = rollback_error


@dataclass(frozen=True)
class _SpeculativeKVBlockState:
    ref_count: int
    generation: int
    block_hash: int
    token_ids: tuple[int, ...]
    was_used: bool


@dataclass(frozen=True)
class _SpeculativeKVHashState:
    primary_block_id: int | None
    block_ids: frozenset[int] | None


@dataclass
class _SpeculativeKVCommitJournal:
    sequence_tables: tuple[tuple[Sequence, tuple[int, ...]], ...]
    transaction_states: tuple[
        tuple[SpeculativeKVTransaction, str], ...
    ]
    blocks: dict[int, _SpeculativeKVBlockState]
    hashes: dict[int, _SpeculativeKVHashState]
    released_block_ids: list[int]
    state: str = "active"
```

`released_block_ids` records only successful free-list appends made by this
commit. It is not a free-list snapshot.

- [ ] **Step 2: Implement bounded capture**

Implement `capture()` using only plan-local IDs:

```python
@classmethod
def capture(cls, manager, plans):
    touched_block_ids = {
        block_id
        for plan in plans
        for block_id in (
            plan.committed_block_ids
            + plan.unused_block_ids
            + tuple(
                publication.block_id
                for publication in plan.publications
            )
        )
    }
    touched_hashes = {
        publication.block_hash
        for plan in plans
        for publication in plan.publications
    }
    blocks = {}
    for block_id in touched_block_ids:
        block = manager.blocks[block_id]
        blocks[block_id] = _SpeculativeKVBlockState(
            ref_count=block.ref_count,
            generation=block.generation,
            block_hash=block.hash,
            token_ids=tuple(block.token_ids),
            was_used=block_id in manager.used_block_ids,
        )
        if block.hash != -1:
            touched_hashes.add(block.hash)
    hashes = {
        block_hash: _SpeculativeKVHashState(
            primary_block_id=manager.hash_to_block_id.get(
                block_hash
            ),
            block_ids=(
                frozenset(manager.hash_to_block_ids[block_hash])
                if block_hash in manager.hash_to_block_ids
                else None
            ),
        )
        for block_hash in touched_hashes
    }
    return cls(
        sequence_tables=tuple(
            (plan.sequence, tuple(plan.sequence.block_table))
            for plan in plans
        ),
        transaction_states=tuple(
            (plan.transaction, plan.transaction.state)
            for plan in plans
        ),
        blocks=blocks,
        hashes=hashes,
        released_block_ids=[],
    )
```

- [ ] **Step 3: Record exact release deltas during apply**

Change `_apply_speculative_kv_commit_plan` to accept the journal:

```python
def _apply_speculative_kv_commit_plan(
    self,
    plan: SpeculativeKVCommitPlan,
    journal: _SpeculativeKVCommitJournal,
) -> None:
    ...
    for block_id in plan.unused_block_ids:
        self.release_reserved_blocks([block_id])
        journal.released_block_ids.append(block_id)
    plan.transaction.state = "committed"
```

Appending to the journal occurs only after `_deallocate_block()` has
successfully appended the ID to `free_block_ids`.

- [ ] **Step 4: Implement reverse-order rollback and terminal states**

Implement:

```python
def rollback(self, manager):
    if self.state != "active":
        raise RuntimeError(
            "speculative KV commit journal is not active: "
            f"{self.state}"
        )
    try:
        for block_id in reversed(self.released_block_ids):
            actual = manager.free_block_ids.pop()
            if actual != block_id:
                raise RuntimeError(
                    "speculative KV free-list rollback order changed"
                )
        for block_id, state in self.blocks.items():
            if state.was_used:
                manager.used_block_ids.add(block_id)
            else:
                manager.used_block_ids.discard(block_id)
        for block_id, state in self.blocks.items():
            block = manager.blocks[block_id]
            block.ref_count = state.ref_count
            block.generation = state.generation
            block.hash = state.block_hash
            block.token_ids = list(state.token_ids)
        for block_hash, state in self.hashes.items():
            if state.block_ids is None:
                manager.hash_to_block_ids.pop(block_hash, None)
            else:
                manager.hash_to_block_ids[block_hash] = set(
                    state.block_ids
                )
            if state.primary_block_id is None:
                manager.hash_to_block_id.pop(block_hash, None)
            else:
                manager.hash_to_block_id[block_hash] = (
                    state.primary_block_id
                )
        for sequence, block_table in self.sequence_tables:
            sequence.block_table = list(block_table)
        for transaction, state in self.transaction_states:
            transaction.state = state
    except BaseException:
        self.state = "rollback_failed"
        raise
    self.state = "rolled_back"
```

The restore order follows the approved design. No loop may use
`for block in manager.blocks`.

- [ ] **Step 5: Replace the full snapshot in batch commit**

After all existing batch prevalidation:

```python
journal = _SpeculativeKVCommitJournal.capture(self, plans)
try:
    for plan in plans:
        self._apply_speculative_kv_commit_plan(plan, journal)
except BaseException as commit_error:
    try:
        journal.rollback(self)
    except BaseException as rollback_error:
        raise SpeculativeKVCommitRollbackError(
            commit_error,
            rollback_error,
        ) from commit_error
    raise
journal.state = "committed"
```

Delete the complete `free_before`, `used_before`, `block_snapshots`, and hash
dictionary snapshots.

- [ ] **Step 6: Run Proposal-KV GREEN and full file**

Run:

```bash
python -m pytest \
  tools/test_speculative_kv_transaction.py \
  -k 'batch and (non_iterable or journal or failure_restores)' \
  -vv
python -m pytest tools/test_speculative_kv_transaction.py -q
```

Expected: all selected tests PASS, then the whole file PASS.

- [ ] **Step 7: Commit the independently reviewable Proposal-KV slice**

Stage only:

```bash
git add \
  tinyvllm/engine/block_manager.py \
  tools/test_speculative_kv_transaction.py
git -c core.hooksPath=/dev/null commit \
  -m "perf: bound proposal KV rollback state" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

Verify the commit contains exactly those two paths and exactly one TRAE CLI
trailer.

---

### Task 3: Establish Scheduler Journal RED Tests

**Files:**
- Modify: `tools/test_scheduler_prepared_postprocess.py:64-83`
- Modify: `tools/test_scheduler_prepared_postprocess.py:192-237`
- Modify: `tools/test_scheduler_prepared_postprocess.py:476-657`
- Modify: `tools/test_engine_speculative_runtime.py:2229-2298`

**Interfaces:**
- Consumes: current `PreparedSchedulerPostprocess.snapshot` and engine
  publication ordering.
- Produces: structural, failure-atomicity, hybrid-release, prefill-hook, and
  rollback-failure tests for `SchedulerPostprocessJournal`.

- [ ] **Step 1: Reuse the non-iterable block guard**

Add the same `_IndexableNonIterableBlocks` implementation used by Task 1 to
this dependency-light test module. Do not import the test helper across test
files.

- [ ] **Step 2: Add scheduler prepare complexity tests**

Create one running sequence in a scheduler with `num_kvcache_blocks=4096`,
replace `scheduler.block_manager.blocks` with the guard after allocation, and
call `prepare_postprocess`.

Assert:

```python
assert isinstance(
    prepared.snapshot,
    scheduler_module.SchedulerPostprocessJournal,
)
assert set(guarded.index_reads).issubset(
    set(sequence.block_table)
)
assert prepared.snapshot.touched_block_count == len(
    sequence.block_table
)
```

Add a second scheduler with the same scheduled sequence size but eight times
the configured capacity; assert the journal entry count is unchanged.

- [ ] **Step 3: Add exact decode and completion rollback injections**

Retain the existing first-row/second-row append failure test and add a
completion test that raises immediately after `_release_request_storage`.
Assert full `_snapshot()` equality, including free-list order and release
events.

For the completion test, patch the next operation (`running.remove`) to raise
after block and optional hybrid lease release.

- [ ] **Step 4: Add prefill publication and hook failure rollback tests**

Use `_scheduled_prefill_sequence` with two full blocks and:

1. patch `_notify_prefill_committed` to raise after
   `block_manager.commit_prefill`;
2. install a hook that raises `RuntimeError("prefill hook failed")`.

Assert both the prefill block hash indexes and all scheduler-owned state equal
the pre-prepare snapshot. For the hook case additionally assert:

```python
assert scheduler._prefill_commit_hook_error == (
    "RuntimeError: prefill hook failed"
)
```

- [ ] **Step 5: Add a real hybrid allocator fixture**

Load `tinyvllm/engine/hybrid_state.py` with torch stubbed only if importing
torch is unavailable, and expose the real `HybridStateLease` and
`HybridStateSlotAllocator` to the scheduler module. Create a completion test
where the scheduled sequence owns a lease, inject after release, and assert:

```python
assert allocator.lease_for_request(sequence.seq_id) == lease_before
assert tuple(allocator._free_slots) == free_slots_before
assert allocator._generations == generations_before
assert tuple(scheduler._hybrid_state_release_events) == events_before
```

The test must also create unrelated leases and assert they are never read or
rewritten by journal capture/rollback.

- [ ] **Step 6: Add Proposal-KV-to-scheduler handoff coverage**

Extend the engine fixture so `prepared_scheduler.snapshot` records a call:

```python
class Journal:
    def extend_speculative_kv_plans(self, scheduler, plans):
        events.append(("scheduler_journal_extend", plans))
```

Assert publication order:

```python
assert events.index(("scheduler_journal_extend", (plan,))) < (
    events.index(("kv_commit",))
)
```

This is required because scheduler prepare precedes Proposal-KV commit and
the bounded scheduler journal must know reserved blocks before they become
sequence-owned.

- [ ] **Step 7: Add scheduler rollback-failure terminal-state coverage**

Patch `prepared.snapshot.rollback` to raise after a scheduler mutation. Assert:

```python
assert prepared.state == "rollback_failed"
with pytest.raises(RuntimeError, match="not active"):
    scheduler.rollback_prepared_postprocess(prepared)
```

At the engine level, inject the same failure and assert:

```python
assert engine.speculative_runtime_poisoned is True
assert "scheduler postprocess rollback failed" in (
    engine.speculative_runtime_poison_reason
)
```

- [ ] **Step 8: Run the scheduler RED slice**

Run:

```bash
python -m pytest \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_engine_speculative_runtime.py \
  -k 'journal or non_iterable or rollback_failure or prefill_hook' \
  -vv
```

Expected: structural tests fail on full block iteration; type and journal
handoff tests fail because `SchedulerPostprocessJournal` and
`extend_speculative_kv_plans()` do not exist.

---

### Task 4: Implement the Bounded Scheduler Postprocess Journal

**Files:**
- Modify: `tinyvllm/engine/scheduler.py:25-43`
- Modify: `tinyvllm/engine/scheduler.py:1288-1543`
- Modify: `tinyvllm/engine/scheduler.py:1675-1941`
- Modify: `tinyvllm/engine/llm_engine.py:262-325`
- Test: `tools/test_scheduler_prepared_postprocess.py`
- Test: `tools/test_engine_speculative_runtime.py`

**Interfaces:**
- Produces:
  - `_SchedulerSequenceState`;
  - `_SchedulerBlockState`;
  - `_SchedulerHashState`;
  - `_SchedulerHybridLeaseState`;
  - `SchedulerPostprocessJournal`;
  - `SchedulerPostprocessJournal.extend_speculative_kv_plans(plans)`;
  - `SchedulerPostprocessJournal.rollback(scheduler)`.
- `PreparedSchedulerPostprocess.snapshot` remains the journal object.

- [ ] **Step 1: Define bounded scheduler journal fields**

Add dataclasses near `PreparedSchedulerPostprocess`:

```python
@dataclass(frozen=True)
class _SchedulerBlockState:
    ref_count: int
    generation: int
    block_hash: int
    token_ids: tuple[int, ...]
    was_used: bool


@dataclass(frozen=True)
class _SchedulerHashState:
    primary_block_id: int | None
    block_ids: frozenset[int] | None


@dataclass
class SchedulerPostprocessJournal:
    sequence_states: tuple[tuple[Sequence, dict], ...]
    waiting: tuple[Sequence, ...]
    prefilling: tuple[Sequence, ...]
    running: tuple[Sequence, ...]
    blocks: dict[int, _SchedulerBlockState]
    hashes: dict[int, _SchedulerHashState]
    original_free_membership: dict[int, bool]
    released_block_ids: list[int]
    hybrid_leases: dict[int, HybridStateLease]
    hybrid_release_event_count: int
    decode_progress: dict[int, tuple[bool, int | None]]
    last_slo_postprocess: dict
    prefill_notified: dict[int, bool]
    prefill_hook_error: object
    adaptive_mixed_state: str
    adaptive_high_streak: int
    adaptive_low_streak: int
    adaptive_consecutive_mixed_steps: int
    consecutive_prefill_chunks: int
    slo_clock_invalid: bool
    slo_clock_invalid_reason: object
    last_slo_decision_now_ns: int | None
    state: str = "active"
```

Populate `original_free_membership` only for touched blocks. Never retain a
complete free-block set.

Expose:

```python
@property
def touched_block_count(self):
    return len(self.blocks)
```

- [ ] **Step 2: Capture scheduled sequence and scalar state**

Replace `_capture_postprocess_snapshot()` with
`_capture_postprocess_journal()` and have `prepare_postprocess()` assign the
result to `snapshot`.

Capture the existing per-sequence fields verbatim, all three queue reference
tuples, and only scheduled IDs from:

```python
decode_progress = {
    seq.seq_id: (
        seq.seq_id in self.decode_progress_ns_by_seq_id,
        self.decode_progress_ns_by_seq_id.get(seq.seq_id),
    )
    for seq in seqs
}
prefill_notified = {
    seq.seq_id: (
        seq.seq_id
        in self._prefill_commit_notified_request_ids
    )
    for seq in seqs
}
```

Copy `_last_slo_postprocess` and scalar controller/clock fields because they
are constant-size transaction authority.

- [ ] **Step 3: Capture only scheduled and predictable publication blocks**

Initialize touched block IDs from scheduled sequence block tables:

```python
touched_block_ids = {
    block_id
    for seq in seqs
    for block_id in seq.block_table
}
```

Capture each block by index. Build touched hash keys from:

- each touched block's current non-negative hash;
- every full prefill block between old/new computed boundaries;
- the computed prefix hash chain for those full blocks.

For each touched hash key preserve only the existing primary ID and duplicate
bucket. Do not copy either complete hash dictionary.

- [ ] **Step 4: Extend the journal with prepared Proposal-KV plans**

Implement:

```python
def extend_speculative_kv_plans(self, scheduler, plans):
    if self.state != "active":
        raise RuntimeError(
            "scheduler postprocess journal is not active: "
            f"{self.state}"
        )
    for plan in plans:
        for block_id in (
            plan.committed_block_ids
            + plan.unused_block_ids
            + tuple(
                publication.block_id
                for publication in plan.publications
            )
        ):
            self._capture_block_if_absent(
                scheduler.block_manager,
                block_id,
            )
        for publication in plan.publications:
            self._capture_hash_if_absent(
                scheduler.block_manager,
                publication.block_hash,
            )
```

In `_commit_prepared_speculative_publication()`, before
`commit_speculative_kv_commit_batch(kv_plans)`, call:

```python
prepared_scheduler.snapshot.extend_speculative_kv_plans(
    engine.scheduler,
    kv_plans,
)
```

This internal handoff keeps `prepare_postprocess()` and
`PreparedSchedulerPostprocess` API-compatible while making rollback cover
Proposal-KV blocks committed between prepare and scheduler commit.

- [ ] **Step 5: Capture hybrid authority only for scheduled requests**

For each scheduled sequence with a valid slot, capture the current lease from
`allocator._request_leases[seq.seq_id]`. Record neither all slots nor all
generation values.

Rollback of a released lease must:

1. remove its slot from the exact free-list append position, requiring the
   expected slot at the deque tail;
2. restore `_owners[slot_id]`;
3. restore `_request_leases[request_id]`;
4. leave the generation unchanged because release does not mutate it;
5. truncate only release events appended after
   `hybrid_release_event_count`.

- [ ] **Step 6: Track block free-list deltas without wrapping global methods**

Before commit, the journal knows touched block membership and current
free-list length. During rollback, for every touched block whose original
membership was used but is now free, require and remove that block from the
free-list suffix in reverse scheduler release order.

Derive release order from the original scheduled sequence block tables:
`BlockManager.deallocate()` releases in reversed table order. Include
Proposal-KV `unused_block_ids` in their plan order. Store this expected order
when extending the journal so rollback never scans the deque for unrelated
IDs.

- [ ] **Step 7: Implement scheduler rollback in approved reverse order**

`SchedulerPostprocessJournal.rollback(scheduler)` must:

1. truncate appended hybrid release events;
2. restore scheduled hybrid leases;
3. remove exact touched block free-list appends;
4. restore touched used membership and block metadata;
5. restore touched hash keys;
6. restore sequence fields;
7. restore waiting/prefilling/running queues;
8. restore selected progress/notification entries and constant-size
   SLO/controller/clock fields.

On success set `state="rolled_back"`. On any exception set
`state="rollback_failed"` and re-raise. Reject any second rollback.

- [ ] **Step 8: Route commit and explicit rollback through the journal**

In `commit_prepared_postprocess()` derive sequences from
`journal.sequence_states`. On commit failure:

```python
commit_error = sys.exc_info()[1]
prefill_hook_error = self._prefill_commit_hook_error
try:
    journal.rollback(self)
except BaseException as rollback_error:
    prepared.state = "rollback_failed"
    raise RuntimeError(
        "scheduler postprocess rollback failed: "
        f"{rollback_error}"
    ) from commit_error
if prefill_hook_error is not None:
    self._prefill_commit_hook_error = prefill_hook_error
prepared.state = "commit_failed"
raise
```

Import `sys` at module scope or use `except BaseException as commit_error`.
Prefer the latter and do not add measured-path introspection.

On success set both `journal.state="committed"` and
`prepared.state="committed"`.

`rollback_prepared_postprocess()` calls `journal.rollback(self)` then sets
`prepared.state="rolled_back"`.

- [ ] **Step 9: Poison runtime on journal rollback failure**

At the outer exception boundary in
`_commit_prepared_speculative_publication()`, before processing other rollback
errors, import `SpeculativeKVCommitRollbackError` from
`tinyvllm.engine.block_manager` and apply:

```python
if prepared_scheduler.state == "rollback_failed":
    engine.speculative_runtime_poisoned = True
    engine.speculative_runtime_poison_reason = (
        "scheduler postprocess rollback failed: "
        f"{error}"
    )
```

If `error` is `SpeculativeKVCommitRollbackError`, use:

```python
engine.speculative_runtime_poisoned = True
engine.speculative_runtime_poison_reason = str(error)
```

Continue existing side-state/finalization rollback attempts, but after they
finish, re-raise the journal rollback failure with the original commit error
as cause. Do not permit later serving with poisoned runtime.

- [ ] **Step 10: Run scheduler and engine GREEN**

Run:

```bash
python -m pytest \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_engine_speculative_runtime.py \
  -k 'journal or non_iterable or rollback_failure or prefill_hook' \
  -vv
python -m pytest \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_engine_speculative_execution.py \
  tools/test_engine_speculative_runtime.py \
  -q
```

Expected: all selected and complete affected files PASS.

- [ ] **Step 11: Commit the scheduler/engine slice**

Stage only:

```bash
git add \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_engine_speculative_runtime.py
git -c core.hooksPath=/dev/null commit \
  -m "perf: bound scheduler rollback state" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

Verify the commit contains exactly these four paths and one TRAE CLI trailer.

---

### Task 5: Run Broad Local Verification and Static Complexity Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: local correctness evidence, a hidden-full-copy audit, and an
  explicit boundary that performance remains unestablished.

- [ ] **Step 1: Run focused transactional suites**

Run:

```bash
python -m pytest \
  tools/test_speculative_kv_transaction.py \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_engine_speculative_execution.py \
  tools/test_engine_speculative_runtime.py \
  -q
```

Expected: PASS. Record exact counts and duration.

- [ ] **Step 2: Run broader affected autoregressive-draft suites**

First list the affected tests without traversing artifact directories:

```bash
rg --files tools \
  | rg 'test_(autoregressive_draft|engine_speculative|scheduler_prepared|speculative_kv)'
```

Run the resulting test files in one `python -m pytest ... -q` command.
Expected: PASS. If an optional dependency prevents collection, report that
as environment setup failure and run the dependency-light files separately;
do not call it a test failure or pass.

- [ ] **Step 3: Run syntax and diff checks without `/tmp`**

Use a task-local cache below the repository:

```bash
PYTHONPYCACHEPREFIX="$PWD/.task-cache/bounded-journal-pycache" \
python -m py_compile \
  tinyvllm/engine/block_manager.py \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_speculative_kv_transaction.py \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_engine_speculative_runtime.py
git diff --check -- \
  tinyvllm/engine/block_manager.py \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_speculative_kv_transaction.py \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_engine_speculative_runtime.py
```

Remove only the named `.task-cache/bounded-journal-pycache` directory after
verification if it is untracked and contains only this task's bytecode.

- [ ] **Step 4: Audit for forbidden full-capacity copies**

Run:

```bash
rg -n \
  'for block in (self|block_manager|manager)\\.blocks|tuple\\([^\\n]*free_block_ids|set\\([^\\n]*used_block_ids|dict\\([^\\n]*hash_to_block' \
  tinyvllm/engine/block_manager.py \
  tinyvllm/engine/scheduler.py
```

Inspect every match. Existing unrelated code may remain, but neither
`commit_speculative_kv_commit_batch()` nor scheduler journal capture/rollback
may contain a full-capacity copy.

Also run:

```bash
rg -n \
  'gc\\.(disable|enable|collect)|\\.item\\(|cuda\\.synchronize|synchronize\\(' \
  tinyvllm/engine/block_manager.py \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/llm_engine.py
```

Expected: no newly added measured-path synchronization or GC control.

- [ ] **Step 5: Reconcile handoff and Phase 1 audit**

Append a dated bounded-journal section to both canonical documents with:

- exact RED commands and failure causes;
- exact GREEN commands and pass counts;
- implementation commit hashes;
- structural proof that journal size scales with touched blocks;
- no claim that Python GC was directly proven;
- no claim that the separate `speculative_prepare` worker/CUDA anomaly was
  fixed;
- `LOCAL_BOUNDED_JOURNAL_CORRECTNESS=ESTABLISHED`;
- `TPOT_TAIL_BENEFIT=NOT_ESTABLISHED` until the paired campaign passes.

- [ ] **Step 6: Commit and push verification documentation**

Stage only:

```bash
git add \
  AGENT_HANDOFF_STATE.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md
git -c core.hooksPath=/dev/null commit \
  -m "docs: record bounded journal verification" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

Verify exactly two paths and one TRAE CLI trailer.

---

### Task 6: Execute the Fresh Source-Bound Paired Performance Gate

**Files:**
- Modify:
  `tools/run_autoregressive_draft_command_timeline_remote.py`
- Create:
  `tools/autoregressive_draft_source_pair_gate.py`
- Create:
  `tools/verify_autoregressive_draft_source_pair_gate.py`
- Create:
  `tools/run_autoregressive_draft_source_pair_remote.py`
- Create:
  `tools/test_autoregressive_draft_source_pair_gate.py`
- Reuse:
  `tools/autoregressive_draft_command_timeline_diagnostic.py`
- Reuse:
  `tools/verify_autoregressive_draft_command_timeline_diagnostic.py`
- Create: fresh local receipt directory only under the repository's existing
  artifact convention if required by the runner.
- Create remotely: a fresh immutable run directory only under
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Modify after results:
  `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`
- Modify after results: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: pinned r23 baseline source archive/bundle and the candidate source
  commit from Tasks 1-5.
- Produces: two independently verified frozen command-timeline bundles, one
  source-pair artifact, two source-pair verifier receipts, and one complete
  source-pair manifest.
- Produces: one of the approved terminal classifications:
  `GO_TPOT_TAIL_OPTIMIZATION`, `NO_GO_TPOT_P95`,
  `NO_GO_TPOT_MEDIAN`, `NO_GO_TTFT_REGRESSION`,
  `NO_GO_THROUGHPUT_REGRESSION`, `NO_GO_CORRECTNESS`,
  `INCONCLUSIVE_STATIONARITY`, `INCONCLUSIVE_ENVIRONMENT`, or
  `INCONCLUSIVE_ARTIFACT`.

- [ ] **Step 1: Add source-pair contract RED tests**

Write focused tests that define:

- exact eight-pair source order and the existing eight-epoch CUDA-mode order;
- four baseline-first and four candidate-first pairs;
- two of each source order inside eager and graph modes;
- 40 measured repeats and 160 request samples per source;
- exact output, Proposal-KV, transaction, and source identity comparison;
- request-level TPOT median/p95 and TTFT p95;
- median batch throughput and fresh-pair regression formulas;
- eager and graph ratio stationarity;
- terminal classification precedence; and
- rejection of missing receipts, mismatched manifests, wrong revisions,
  incomplete ranks, non-finite metrics, and verifier disagreement.

Run the new test file and preserve failure because the source-pair modules do
not yet exist.

- [ ] **Step 2: Implement the pure source-pair artifact and verifier**

`autoregressive_draft_source_pair_gate.py` must be dependency-light and pure:
it loads the two verified command-timeline artifacts plus their normalized
receipts and builds a canonical source-pair artifact. It must not launch a
worker or mutate either source bundle.

`verify_autoregressive_draft_source_pair_gate.py` independently reloads the
bound artifacts, verifies their hashes/manifests/receipts, rebuilds the
source-pair artifact, verifies the complete source-pair manifest when
provided, and emits an exclusive receipt.

Run the focused RED tests to GREEN before editing orchestration.

- [ ] **Step 3: Add revision archive and orchestration RED tests**

Extend the focused tests to require:

- Git-object export for an arbitrary full 40-hex revision without a checkout;
- no local patch applied to either frozen source;
- baseline revision fixed to
  `596e724ea87966b2ab3b47cccda08c106f9084bb`;
- candidate revision equal to local HEAD and
  `origin/feat/kv-sparse-attention`;
- parent, baseline, candidate, primary-comparison, and
  controller-comparison paths all below the Sitian task root;
- no destination overwrite;
- Kerberos TTL fail-fast before remote mutation;
- strict exactly-four clean-GPU admission;
- before/after frozen inventory around every pair member;
- pair-member order matching the frozen source-pair schedule;
- partial-copy preservation on failure without adopting or signalling
  unrelated processes; and
- source-bundle finalization before source-pair comparison.

Run the focused tests and preserve the expected missing-orchestration
failures.

- [ ] **Step 4: Implement source-version paired orchestration**

`run_autoregressive_draft_source_pair_remote.py` imports the established
command-timeline runner and reuses its prepare, inventory, epoch, assemble,
manifest, verifier, controller-copy, and receipt-comparison actions.

It exports the baseline and candidate from exact Git revisions, prepares two
immutable child tags, and interleaves corresponding epochs as:

```text
pair 0: eager  baseline -> candidate
pair 1: graph  candidate -> baseline
pair 2: graph  baseline -> candidate
pair 3: eager  candidate -> baseline
pair 4: graph  baseline -> candidate
pair 5: eager  baseline -> candidate
pair 6: eager  candidate -> baseline
pair 7: graph  candidate -> baseline
```

Each child retains five measured repeats per epoch. After both child bundles
pass their existing dual verification, the candidate frozen source builds and
verifies the parent source-pair artifact in primary and controller locations.

Add the three new source-pair files to the current candidate source archive
inventory in
`tools/run_autoregressive_draft_command_timeline_remote.py`. Baseline export
continues to use the exact paths present in the pinned revision.

All task-controlled cache, pycache, XDG, logs, receipts, manifests, and
scratch paths remain below:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
```

Run the focused orchestration tests to GREEN, then run the existing
command-timeline runner, diagnostic, and verifier suites.

- [ ] **Step 5: Reconfirm campaign authority and source identity**

Before any remote launch, record:

```bash
git rev-parse HEAD
git status --short -- \
  tinyvllm/engine/block_manager.py \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_speculative_kv_transaction.py \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_engine_speculative_runtime.py
```

Candidate source must be committed and clean on these exact paths. Baseline
must resolve to the preserved source behavior for:

```text
596e724ea87966b2ab3b47cccda08c106f9084bb
```

Do not create a worktree. Package baseline and candidate from exact source
archives/bundles.

- [ ] **Step 6: Run read-only remote admission**

Use the established remote preflight and require exactly four rows satisfying:

```text
memory.used <= 1024 MiB
utilization.gpu <= 5%
no compute processes
```

Do not signal or adopt unrelated processes. If admission fails, preserve the
receipt and classify `INCONCLUSIVE_ENVIRONMENT` with exit code `2`.

- [ ] **Step 7: Launch one fresh immutable paired tag**

Use a never-before-attempted tag and preserve:

```text
TP = 4
batch = 4
Q = 4
prompt tokens = 256
output tokens = 16
temperature = 0
Proposal-KV allocator = direct
Proposal-KV offload = disabled
balanced eager/graph order
strict four-clean-GPU admission
dual verification
complete manifest
stationarity requirement
```

All environment variables that control cache, pycache, XDG, logs, and scratch
must point below the Sitian task root before Python starts.

- [ ] **Step 8: Verify correctness and performance independently**

Require:

```text
exact output tokens:           pass
Proposal-KV transactions:      pass
four-rank correctness:         pass
paired stationarity:           pass
overall TPOT p95:              <= 105.87 ms
overall TPOT median:           <= 85.66 ms
TTFT p95 regression:           <= 3%
throughput regression:         <= 3%
```

Run both existing independent verifiers against the complete manifest. A
missing row, mismatched source hash, incomplete rank, or verifier disagreement
is `INCONCLUSIVE_ARTIFACT`, not a performance result.

- [ ] **Step 9: Reconcile the final classification**

Update the audit and handoff with:

- immutable tag;
- baseline and candidate commit/source hashes;
- GPU admission receipt;
- exact paired schedule;
- all aggregate metrics and regression percentages;
- correctness and stationarity outcomes;
- both verifier outputs;
- the single terminal classification;
- residual evidence for the separate worker/CUDA anomaly, if observed.

Only `GO_TPOT_TAIL_OPTIMIZATION` permits the claim that this optimization
reduced TPOT tail latency under the fixed gate.

- [ ] **Step 10: Commit and push the final receipts and reconciliation**

Use exact-path staging for the audit, handoff, manifest, verifier receipts,
and any intentionally tracked compact result files. Do not stage raw unrelated
artifacts.

Commit with:

```bash
git -c core.hooksPath=/dev/null commit \
  -m "perf: record bounded journal paired gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

Verify the pushed commit contains exactly one TRAE CLI trailer and that
`origin/feat/kv-sparse-attention` resolves to the local HEAD.
