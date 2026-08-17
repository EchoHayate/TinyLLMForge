# Generic Speculative KV Transaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a model-independent, fail-closed speculative KV append transaction with generation-safe reservation, explicit materialization, and exactly-once commit or rollback.

**Architecture:** `BlockManager` owns a small transaction dataclass and all lifecycle transitions. The transaction snapshots logical request ownership and private reserved block generations; the execution layer explicitly acknowledges materialized KV before acceptance metadata can commit.

**Tech Stack:** Python 3, dataclasses, existing `BlockManager` and `Sequence`, CPU-only pytest-compatible tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, or run `git clean`.
- Keep runtime core model-independent; do not add model-name branches.
- Preserve existing append reservation and commit APIs in this phase.
- Do not integrate scheduler batching, MTP adapters, CUDA Graphs, or KV offload.
- Do not simulate H2D copies or claim physical KV-offload savings.
- Accepted KV must commit without token-by-token rematerialization; rejected suffix ownership must roll back.
- All production behavior must be introduced through a witnessed RED test first.
- Do not claim latency, throughput, memory, parity, or acceptance improvements from this CPU contract.

---

### Task 1: Transaction Reservation and Materialization

**Files:**
- Create: `tools/test_speculative_kv_transaction.py`
- Modify: `tinyvllm/engine/block_manager.py`

**Interfaces:**
- Consumes: `Sequence.seq_id`, `Sequence.block_table`, `Block.generation`.
- Produces: `SpeculativeKVTransaction`.
- Produces: `BlockManager.begin_speculative_kv_transaction()`.
- Produces: `BlockManager.mark_speculative_kv_materialized()`.

- [x] **Step 1: Write failing reservation tests**

Add tests that construct real CPU `Sequence` and `BlockManager` instances and
assert:

```python
transaction = manager.begin_speculative_kv_transaction(
    sequence,
    proposed_token_count=5,
)
assert transaction.sequence_id == sequence.seq_id
assert transaction.original_num_tokens == len(sequence)
assert transaction.original_block_table == tuple(sequence.block_table)
assert transaction.proposed_token_count == 5
assert transaction.state == "reserved"
assert sequence.block_table == original_table
```

Cover no-extra-block, one-extra-block, insufficient-capacity, invalid count,
and generation capture cases.

- [x] **Step 2: Run reservation tests and verify RED**

Run:

```bash
python3 -m pytest -q tools/test_speculative_kv_transaction.py -k "begin or capacity"
```

Expected: collection or assertion failure because
`SpeculativeKVTransaction` and
`begin_speculative_kv_transaction()` do not exist.

- [x] **Step 3: Implement minimal reservation**

Add:

```python
@dataclass
class SpeculativeKVTransaction:
    sequence_id: int
    original_num_tokens: int
    original_last_token: int
    original_block_table: tuple[int, ...]
    reserved_block_ids: tuple[int, ...]
    reserved_block_generations: tuple[int, ...]
    proposed_token_count: int
    materialized_token_count: int = 0
    state: str = "reserved"
```

Implement `begin_speculative_kv_transaction()` with complete prevalidation,
`N - 1` verifier-visible capacity calculation, partial-allocation cleanup,
generation capture, and no sequence mutation.

- [x] **Step 4: Run reservation tests and verify GREEN**

Run the focused command from Step 2.

Expected: all selected tests pass.

- [x] **Step 5: Write failing materialization tests**

Assert:

```python
manager.mark_speculative_kv_materialized(transaction, 4)
assert transaction.materialized_token_count == 4
assert transaction.state == "materialized"
```

Also cover zero, negative, above `proposed_token_count - 1`, repeated
acknowledgement, malformed transaction, and stale generation.

- [x] **Step 6: Run materialization tests and verify RED**

Run:

```bash
python3 -m pytest -q tools/test_speculative_kv_transaction.py -k materialized
```

Expected: failure because
`mark_speculative_kv_materialized()` does not exist.

- [x] **Step 7: Implement minimal materialization transition**

Validate transaction structure, state, block IDs, generations, live ownership,
refcount `1`, unpublished hashes, and count bounds before setting:

```python
transaction.materialized_token_count = materialized_token_count
transaction.state = "materialized"
```

- [x] **Step 8: Run focused tests and verify GREEN**

Run the focused command from Step 6.

Expected: all selected tests pass.

### Task 2: Commit and Rollback

**Files:**
- Modify: `tools/test_speculative_kv_transaction.py`
- Modify: `tinyvllm/engine/block_manager.py`

**Interfaces:**
- Consumes: `SpeculativeKVTransaction` from Task 1.
- Produces: `BlockManager.commit_speculative_kv_transaction()`.
- Produces: `BlockManager.rollback_speculative_kv_transaction()`.

- [x] **Step 1: Write failing commit tests**

Cover zero, one, partial, and full acceptance. For each case assert exact:

```python
sequence.token_ids
sequence.block_table
manager.free_block_ids
transaction.state
```

Add fail-closed cases for wrong sequence, sequence token/block-table drift,
stale generation, accepted count above proposal, accepted materialized prefix
above acknowledgement, boolean/non-integer accepted tokens, and commit before
materialization.

- [x] **Step 2: Run commit tests and verify RED**

Run:

```bash
python3 -m pytest -q tools/test_speculative_kv_transaction.py -k commit
```

Expected: failure because `commit_speculative_kv_transaction()` does not
exist.

- [x] **Step 3: Implement prevalidated commit**

Before mutation, validate owner/snapshot/state, all original and reserved block
ownership, accepted token types/counts, materialized coverage, and final block
capacity. Validate the accepted materialized publication range before the
commit point.

After validation:

```python
sequence.block_table.extend(committed_reserved_blocks)
for token_id in accepted_tokens:
    sequence.append_token(token_id)
manager.publish_full_blocks(
    sequence,
    materialized_tokens=materialized_end,
)
manager.release_reserved_blocks(unused_reserved_blocks)
transaction.state = "committed"
```

Zero acceptance releases every reserved block without mutating the sequence.

- [x] **Step 4: Run commit tests and verify GREEN**

Run the focused command from Step 2.

Expected: all selected tests pass.

- [x] **Step 5: Write failing rollback tests**

Cover rollback from `reserved` and `materialized`, same-owner sequence drift,
wrong owner, stale generation, rollback after commit, double rollback, and
commit after rollback.

- [x] **Step 6: Run rollback tests and verify RED**

Run:

```bash
python3 -m pytest -q tools/test_speculative_kv_transaction.py -k rollback
```

Expected: failure because `rollback_speculative_kv_transaction()` does not
exist.

- [x] **Step 7: Implement exactly-once rollback**

Validate owner, state, and private reserved ownership. Release all reserved
blocks without touching sequence tokens or its block table, then set:

```python
transaction.state = "rolled_back"
```

- [x] **Step 8: Run rollback tests and verify GREEN**

Run the focused command from Step 6.

Expected: all selected tests pass.

### Task 3: Compatibility and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-08-12-generic-speculative-kv-transaction.md`

**Interfaces:**
- Verifies: new transaction contract against existing block-manager and native speculative behavior.
- Records: next integration gate without performance overclaim.

- [x] **Step 1: Run the complete new contract matrix**

Run:

```bash
python3 -m pytest -q tools/test_speculative_kv_transaction.py
```

Expected: all tests pass.

- [x] **Step 2: Run existing model-independent regressions**

Run:

```bash
python3 -m pytest -q tools/test_ngram_speculative.py
python3 -m pytest -q tools/test_chunked_prefill.py
python3 -m pytest -q tools/test_model_runner_spec_verify.py
```

Expected: all available tests pass; any environment-only skip or missing
optional dependency is reported separately and not described as a pass.

- [x] **Step 3: Run syntax and diff hygiene**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/block_manager.py \
  tools/test_speculative_kv_transaction.py
git diff --check
```

Expected: both commands exit zero.

- [x] **Step 4: Update handoff**

Record:

- the exact transaction API and state machine;
- focused and regression test evidence;
- that no scheduler, model adapter, GPU, TP, long-context, or performance
  claim was made;
- the next gate: migrate native `verify_and_commit_block()` from bare block
  IDs to the transaction, then move the source-agnostic orchestration into
  runtime code before adding batch scheduler support.

- [x] **Step 5: Self-audit completion**

Confirm every requirement in the design maps to a passing test, scan for
placeholders or model-name branches, and mark plan checkboxes only from fresh
command output.

## Fresh Completion Evidence

```text
python3 -m pytest -q tools/test_speculative_kv_transaction.py
25 passed

python3 -m pytest -q tools/test_ngram_speculative.py
59 passed

PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.12 tools/test_chunked_prefill.py
chunked prefill tests passed

python3 -m py_compile ...
PASS

/opt/homebrew/bin/python3.12 -m py_compile ...
PASS

git diff --check
PASS

git diff --cached --name-only
empty
```

`tools/test_model_runner_spec_verify.py` did not collect under the system
Python because its dependency-light source loader does not install the
existing `tinyvllm.engine.decode_internal_profiler` module imported by
`model_runner.py`. This is a test-harness blocker, not a transaction test
failure, and is not reported as a pass.
