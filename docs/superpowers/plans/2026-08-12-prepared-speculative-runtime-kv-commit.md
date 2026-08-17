# Prepared Speculative Runtime KV Commit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split generic speculative execution into a non-committing prepare
phase and an all-or-nothing, token-free KV ownership commit that can be safely
owned by a later engine transaction.

**Architecture:** `prepare_native_speculative_batch()` performs target
callbacks, proposals, reservations, materialization, and acceptance while
keeping every transaction private. BlockManager converts prepared rows into
fully validated immutable commit plans, then applies the whole batch with
rollback snapshots. The legacy execute API remains a compatibility wrapper
that appends accepted tokens only after the KV batch commit succeeds.

**Tech Stack:** Python 3.9+, dataclasses, dependency-light pytest, existing
BlockManager speculative transactions.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, or run
  `git clean`.
- Keep generic runtime/core free of model-name and proposal-source branches.
- Accepted KV must be committed directly; rejected suffix blocks must be
  released without token replay or KV copy.
- No sequence token or live block-table mutation is allowed during prepare.
- No performance, GPU parity, TP1/TP4, or promotion claim is allowed.
- Preserve `execute_native_speculative_batch()` for existing profiler callers.

---

### Task 1: Prepared Runtime RED Contract

**Files:**
- Modify: `tools/test_speculative_batch_runtime.py`
- Modify: `tools/test_speculative_public_api.py`

**Interfaces:**
- Consumes: existing `execute_native_speculative_batch()` fixtures.
- Produces: failing contracts for
  `PreparedNativeSpeculativeSequence`,
  `PreparedNativeSpeculativeBatch`,
  `prepare_native_speculative_batch()`, and
  `rollback_prepared_native_speculative_batch()`.

- [x] **Step 1: Add prepared-result public API assertions**

Require the four names above from `tinyvllm.speculative`.

- [x] **Step 2: Add prepare-without-commit test**

Use the existing batch-4 fixture and assert:

```python
prepared = prepare_native_speculative_batch(
    block_manager=block_manager,
    seqs=seqs,
    draft_adapter=adapter,
    eos_token=2,
    run_first_targets=callbacks.run_first_targets,
    run_tail_batch=callbacks.run_tail_batch,
)

assert tuple(seq.token_ids for seq in seqs) == before_tokens
assert tuple(seq.block_table for seq in seqs) == before_block_tables
assert all(
    row.transaction is None
    or row.transaction.state == "materialized"
    for row in prepared.sequences
)
```

- [x] **Step 3: Add prepared rollback test**

Call `rollback_prepared_native_speculative_batch()` and require every
transaction to become `rolled_back`, every reserved block to be released, and
all sequence token/block snapshots to remain unchanged. Repeated rollback must
fail closed.

- [x] **Step 4: Run RED**

```bash
python3 -m pytest -q \
  tools/test_speculative_batch_runtime.py \
  tools/test_speculative_public_api.py
```

Expected: collection or assertions fail because the prepared API does not
exist.

---

### Task 2: Implement Prepared Runtime

**Files:**
- Modify: `tinyvllm/speculative/batch_runtime.py`
- Modify: `tinyvllm/speculative/__init__.py`

**Interfaces:**
- Consumes: existing callback/result validation and transaction reservation
  logic.
- Produces:

```python
@dataclass(frozen=True)
class PreparedNativeSpeculativeSequence:
    sequence_id: int
    sequence: object
    first_target_token: int
    proposal: DraftProposal
    plan: SpecVerifyPlan | None
    target_tokens: tuple[int, ...]
    greedy_accepted_count: int
    accepted_tokens: tuple[int, ...]
    eos_truncated: bool
    output_budget_truncated: bool
    transaction: object | None
    reserved_blocks: tuple[int, ...]
    proxy_block_table: tuple[int, ...]
    first_target_metadata: object | None = None
    tail_metadata: object | None = None
    tail_auxiliary: object | None = None


@dataclass
class PreparedNativeSpeculativeBatch:
    sequences: tuple[PreparedNativeSpeculativeSequence, ...]
    first_target_callback_count: int
    tail_callback_count: int
    timing_ms: dict[str, float]
    state: str = "prepared"
```

- [x] **Step 1: Extract prepare phase**

Move existing phases through acceptance into
`prepare_native_speculative_batch()`. Store transactions on prepared rows and
return before the current commit loop.

- [x] **Step 2: Add exact prepared validation**

Validate unique ordered IDs, row/sequence identity, transaction ownership,
accepted proposal prefix, and terminal state. A row with an empty proposal
must carry `transaction=None`.

- [x] **Step 3: Implement rollback**

`rollback_prepared_native_speculative_batch()` rolls back every non-`None`
materialized transaction in sequence order. It changes the batch state to
`rolled_back` only after every rollback succeeds; aggregate rollback failures
in `NativeSpeculativeBatchError`.

- [x] **Step 4: Export the prepared API**

Update `tinyvllm/speculative/__init__.py` and `__all__`.

- [x] **Step 5: Run GREEN**

Run the Task 1 command and require all tests to pass.

---

### Task 3: KV Commit Plan RED Contract

**Files:**
- Modify: `tools/test_speculative_kv_transaction.py`

**Interfaces:**
- Consumes: materialized transactions from existing fixtures.
- Produces: failing contracts for:

```python
SpeculativeKVCommitPlan
BlockManager.prepare_speculative_kv_commit(...)
BlockManager.commit_speculative_kv_commit_batch(...)
```

- [x] **Step 1: Add non-mutating plan test**

Prepare a partial-accept transaction and assert plan creation leaves:

```text
Sequence.token_ids unchanged
Sequence.block_table unchanged
transaction.state == materialized
allocator/hash indexes unchanged
```

Assert the plan records exact committed/unused block IDs, materialized end,
accepted tokens, and full-block publication rows.

- [x] **Step 2: Add token-free batch commit test**

Commit two plans and assert:

- accepted block ownership becomes live;
- unused reservations are released;
- prefix-cache publication matches
  `original_token_ids + accepted_tokens`;
- `Sequence.token_ids` and `last_token` are unchanged;
- only `Sequence.block_table` changes;
- both transactions become `committed`.

- [x] **Step 3: Add all-or-nothing failure matrix**

Inject failure at each plan index and require complete restoration of:

```text
sequence block tables
free and used block sets
block refcounts/generations/hashes/token metadata
hash indexes
transaction states
```

- [x] **Step 4: Run RED**

```bash
python3 -m pytest -q tools/test_speculative_kv_transaction.py
```

Expected: the plan type and methods are absent.

---

### Task 4: Implement Token-Free Atomic KV Commit

**Files:**
- Modify: `tinyvllm/engine/block_manager.py`

**Interfaces:**
- Consumes: materialized `SpeculativeKVTransaction` plus accepted token tuple.
- Produces:

```python
@dataclass(frozen=True)
class SpeculativeKVCachePublication:
    block_id: int
    block_hash: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class SpeculativeKVCommitPlan:
    sequence_id: int
    sequence: Sequence
    transaction: SpeculativeKVTransaction
    accepted_tokens: tuple[int, ...]
    committed_block_ids: tuple[int, ...]
    unused_block_ids: tuple[int, ...]
    materialized_end: int
    publications: tuple[SpeculativeKVCachePublication, ...]
```

- [x] **Step 1: Implement non-mutating planner**

Reuse transaction structure/owner/block validation. Compute block ownership
and cache hashes from:

```python
planned_token_ids = (
    tuple(seq.token_ids)
    + accepted_tokens
)
```

Reject duplicate transactions/sequences and stale snapshots before mutation.

- [x] **Step 2: Snapshot the batch mutation surface**

Capture free/used ownership, affected blocks, hash indexes, sequence block
tables, and transaction states for every plan.

- [x] **Step 3: Apply token-free plans**

Extend only sequence block tables, publish precomputed full-block rows, release
unused reservations, and set transaction state to `committed`. Do not call
`seq.append_token()`.

- [x] **Step 4: Restore on any failure**

On exception restore every captured surface and re-raise. No transaction may
remain committed after a failed batch.

- [x] **Step 5: Run GREEN**

Run the Task 3 command and require all tests to pass.

---

### Task 5: Prepared Commit and Legacy Wrapper

**Files:**
- Modify: `tinyvllm/speculative/batch_runtime.py`
- Modify: `tools/test_speculative_batch_runtime.py`

**Interfaces:**
- Consumes:
  `PreparedNativeSpeculativeBatch` and BlockManager KV commit plans.
- Produces:

```python
commit_prepared_native_speculative_batch(
    *,
    block_manager,
    prepared: PreparedNativeSpeculativeBatch,
) -> NativeSpeculativeBatchResult
```

- [x] **Step 1: Add commit RED tests**

Require commit to:

- prepare all KV plans before mutation;
- call one atomic BlockManager batch commit;
- return committed/released block IDs;
- leave sequence tokens unchanged;
- reject rolled-back or already-committed prepared batches.

- [x] **Step 2: Implement prepared commit**

Build every plan, commit the batch, create
`NativeSpeculativeSequenceResult` rows, and move prepared state from
`prepared` to `committed`.

- [x] **Step 3: Preserve legacy execute behavior**

Implement `execute_native_speculative_batch()` as:

```python
prepared = prepare_native_speculative_batch(...)
result = commit_prepared_native_speculative_batch(
    block_manager=block_manager,
    prepared=prepared,
)
for row in result.sequences:
    seq = sequence_by_id[row.sequence_id]
    for token_id in row.accepted_tokens:
        seq.append_token(token_id)
return result
```

If token append fails, raise a poisoned compatibility-wrapper error rather
than claiming rollback of already committed KV.

- [x] **Step 4: Replace partial-commit expectations**

Existing tests that accepted `committed_sequence_ids == (1,)` must now require
zero committed rows and full rollback for a commit-batch failure.

- [x] **Step 5: Run GREEN**

```bash
python3 -m pytest -q \
  tools/test_speculative_batch_runtime.py \
  tools/test_speculative_kv_transaction.py \
  tools/test_speculative_public_api.py
```

---

### Task 6: Regression and Evidence

**Files:**
- Modify:
  `docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/plans/2026-08-12-prepared-speculative-runtime-kv-commit.md`

**Interfaces:**
- Consumes: completed prepared runtime and atomic KV batch commit.
- Produces: fresh evidence and the explicit next engine/scheduler wiring gate.

- [x] **Step 1: Run focused regression**

Run the current ModelRunner/speculative/engine matrix, including the sequence
temperature tests, native verifier attention script, and chunked-prefill
script.

- [x] **Step 2: Run compatibility and hygiene**

Run Python 3.9/3.12 `py_compile`, generic source scan, `git diff --check`,
unchecked-plan scan, and staged-diff-empty validation.

- [x] **Step 3: Update evidence**

Record:

```text
prepared speculative runtime:
  implemented; no live Sequence token/block-table mutation
atomic token-free KV batch commit:
  implemented
LLMEngine selected/suppressed execution:
  not implemented
Scheduler multi-token postprocess:
  not implemented
overall classification:
  NOT_PROMOTABLE
```

- [x] **Step 4: Complete the plan**

Only after fresh verification, change every checkbox to `[x]`.

## Fresh Completion Evidence

```text
prepared runtime RED:
  collection failed because prepared API was absent
prepared runtime focused GREEN:
  18 passed
KV commit plan RED:
  3 failed because plan/batch commit APIs were absent
prepared/KV/public API focused GREEN:
  47 passed
ModelRunner/speculative/engine/serialization regression:
  320 passed
native verifier attention:
  passed; CUDA numerical cases deferred to remote gate
chunked prefill:
  passed
Python 3.9 and 3.12 py_compile:
  passed
generic source scan and git diff hygiene:
  passed; staged diff empty
```

Strict boundary:

```text
prepared speculative runtime:
  implemented; no live Sequence token/block-table mutation
atomic token-free KV batch commit:
  implemented with full allocator/hash rollback snapshots
legacy execute wrapper:
  preserved; appends accepted tokens only after KV batch commit
LLMEngine selected/suppressed execution:
  not implemented
Scheduler multi-token postprocess:
  not implemented
overall classification:
  NOT_PROMOTABLE
```
