# Generic Proposal-KV Lifecycle Coordinator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract proposal-KV registration, batched finalize, rollback, and sequence release into a model-independent coordinator, then migrate Qwen3.5 native MTP to it without behavior change.

**Architecture:** Add a dependency-light `ProposalKVLifecycleCoordinator` over the existing `ProposalKVCache`. The coordinator owns transaction registration and finalize tickets; `Qwen35MTPProposalExecutor` keeps only model-specific prefill/bootstrap/proposal execution and delegates lifecycle operations.

**Tech Stack:** Python dataclasses, existing `ProposalKVCache`, `DraftProposal`, dependency-light pytest.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not create or switch worktrees or branches.
- Do not stage, commit, push, stash, reset, or clean.
- Do not run remote or GPU workloads.
- Preserve exact greedy parity and `MAX_PROPOSAL_TOKENS=4`.
- Do not change target-KV transactions, verifier selection, fallback indexing, accepted-prefix semantics, recurrent side-state selection, or offload counters.
- Do not claim a second learned structure, KV8/KV4, performance improvement, production readiness, or Phase 1 completion.
- Every production change follows RED -> GREEN with the focused test observed failing first.

---

### Task 1: Generic Registration Owner

**Files:**
- Create: `tinyvllm/engine/proposal_kv_lifecycle.py`
- Create: `tools/test_proposal_kv_lifecycle.py`

**Interfaces:**
- Consumes: `ProposalKVCache.transaction()`, `ProposalKVCache.abort()`, `DraftProposal`.
- Produces: `ProposalKVRegistration` and `ProposalKVLifecycleCoordinator.register_batch()`.

- [x] **Step 1: Write failing registration tests**

Create fixtures with a dependency-light physical slot store and real
`ProposalKVCache`. Cover stable mixed empty/non-empty output, transaction
identity validation, duplicate identities, and batch cleanup:

```python
def test_register_batch_preserves_order_and_tracks_materialized_transactions():
    cache, first, second = materialized_transactions()
    coordinator = ProposalKVLifecycleCoordinator(
        cache,
        ticket_namespace="fixture",
    )
    empty = DraftProposal(
        sequence_id=2,
        token_ids=(),
        source_type="learned",
    )
    proposed = DraftProposal(
        sequence_id=1,
        token_ids=(11, 12),
        source_type="learned",
        proposal_transaction_id=first.transaction_id,
    )

    rows = coordinator.register_batch((
        ProposalKVRegistration(2, 0, empty),
        ProposalKVRegistration(1, 0, proposed),
    ))

    assert rows == (empty, proposed)
    assert coordinator.active_transaction_count == 1
```

Add a failure case where the second row has a stale epoch and assert the first
new transaction is aborted while a transaction registered by an earlier
successful batch remains active.

- [x] **Step 2: Run registration tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD /tmp/tinyllmforge-trace-test-venv/bin/python \
  -m pytest -q tools/test_proposal_kv_lifecycle.py
```

Expected: collection fails because
`tinyvllm.engine.proposal_kv_lifecycle` does not exist.

- [x] **Step 3: Implement the registration owner**

Create:

```python
@dataclass(frozen=True)
class ProposalKVRegistration:
    sequence_id: int
    sequence_epoch: int
    proposal: DraftProposal


class ProposalKVLifecycleCoordinator:
    def __init__(self, proposal_kv_cache, *, ticket_namespace):
        self.proposal_kv_cache = proposal_kv_cache
        self.ticket_namespace = normalize_namespace(ticket_namespace)
        self._active_transactions = {}
        self._batch_tickets = {}
        self._next_batch_ticket_id = 1
        self._authority_rows = {}
        self._release_rows = []

    def register_batch(self, rows):
        ...
```

Implement validation from the design. Register only non-empty proposals.
Abort only new unregistered transactions on batch failure, in reverse row
order.

- [x] **Step 4: Run registration tests and verify GREEN**

Run the Task 1 command. Expected: all Task 1 tests pass.

---

### Task 2: Generic Finalize and Release

**Files:**
- Modify: `tinyvllm/engine/proposal_kv_lifecycle.py`
- Modify: `tools/test_proposal_kv_lifecycle.py`

**Interfaces:**
- Consumes: active registrations from Task 1 and
  `ProposalFinalizeRow`.
- Produces: `prepare_finalize_batch()`, `commit_finalize_batch()`,
  `rollback_finalize_batch()`, `assert_sequence_releasable()`,
  `release_sequence()`, and `authority_snapshot()`.

- [x] **Step 1: Write failing finalize and release tests**

Add tests proving:

```python
ticket_id = coordinator.prepare_finalize_batch((
    ProposalFinalizeRow(
        sequence_id=1,
        proposal_transaction_id=transaction.transaction_id,
        accepted_proposal_tokens=2,
    ),
))
assert ticket_id == "fixture-finalize-1"
coordinator.commit_finalize_batch(ticket_id)
assert cache.committed_length(1) == 1
assert coordinator.active_transaction_count == 0
```

Also cover:

- partial prepare failure rolls prior underlying tickets back, aborts
  remaining materialized transactions, and removes the whole failed batch
  from active ownership;
- rollback uses reverse order;
- batch tickets are single-use;
- active transactions block release;
- stale epochs block release;
- committed slots are released on sequence release;
- authority snapshots contain plain tensor-free values.

- [x] **Step 2: Run new tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD /tmp/tinyllmforge-trace-test-venv/bin/python \
  -m pytest -q tools/test_proposal_kv_lifecycle.py
```

Expected: failures for missing finalize/release APIs.

- [x] **Step 3: Implement finalize and release**

Use a private batch ticket record:

```python
@dataclass(frozen=True)
class _ProposalKVBatchFinalize:
    underlying_ticket_ids: tuple[str, ...]
    transaction_ids: tuple[str, ...]
```

`prepare_finalize_batch()` publishes the coordinator ticket only after every
underlying ticket is prepared. On partial failure it rolls prepared tickets
back, aborts the unprepared materialized suffix, removes the requested
registrations, and records terminal cleanup evidence.
`commit_finalize_batch()` commits in row order; `rollback_finalize_batch()`
rolls back in reverse order. Both consume their coordinator ticket exactly
once.

Expose:

```python
@property
def active_transaction_count(self) -> int: ...

@property
def prepared_ticket_count(self) -> int: ...

def authority_snapshot(self) -> dict: ...
```

- [x] **Step 4: Run coordinator tests and verify GREEN**

Run the Task 2 command. Expected: all coordinator tests pass.

---

### Task 3: Migrate Qwen3.5 Native MTP

**Files:**
- Modify: `tinyvllm/engine/qwen35_mtp_executor.py`
- Modify: `tools/test_qwen35_mtp_executor.py`

**Interfaces:**
- Consumes: `ProposalKVLifecycleCoordinator` from Tasks 1-2.
- Produces: unchanged `Qwen35MTPProposalExecutor` public behavior and snapshot
  schema.

- [x] **Step 1: Write failing delegation tests**

Add a focused assertion that the executor owns a coordinator and that proposal
registration/finalization passes through it:

```python
assert isinstance(
    executor.proposal_kv_lifecycle,
    ProposalKVLifecycleCoordinator,
)
proposal = executor.propose_batch((input_row,))[0]
assert executor.proposal_kv_lifecycle.active_transaction_count == 1
ticket = executor.prepare_finalize_batch((
    ProposalFinalizeRow(
        sequence_id=input_row.sequence_id,
        proposal_transaction_id=proposal.proposal_transaction_id,
        accepted_proposal_tokens=1,
    ),
))
executor.commit_finalize_batch(ticket)
assert executor.proposal_kv_lifecycle.active_transaction_count == 0
```

Retain existing snapshot equality assertions.

- [x] **Step 2: Run Qwen3.5 executor tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD /tmp/tinyllmforge-trace-test-venv/bin/python \
  -m pytest -q tools/test_qwen35_mtp_executor.py
```

Expected: failure because `proposal_kv_lifecycle` is absent.

- [x] **Step 3: Delegate lifecycle operations**

Construct:

```python
self.proposal_kv_lifecycle = ProposalKVLifecycleCoordinator(
    proposal_kv_cache,
    ticket_namespace="qwen35-mtp",
)
```

Replace `_register_group_proposals()` with registration rows built from
`input_row.sequence_id`, `bootstrap.sequence_epoch`, and each proposal.
Delegate prepare/commit/rollback finalize.

For release:

```python
self.proposal_kv_lifecycle.assert_sequence_releasable(
    sequence_id,
    sequence_epoch,
)
# validate pending/bootstrap Qwen3.5 state
self.proposal_kv_lifecycle.release_sequence(
    sequence_id,
    sequence_epoch,
)
```

Build `tp4_authority_snapshot()` from the coordinator snapshot while retaining
the exact current keys:

```text
proposal_transactions
release_rows
active_transactions
prepared_tickets
allocated_physical_slots
proposal_kv_cache
```

Delete only lifecycle fields and helpers made redundant in
`Qwen35MTPProposalExecutor`.

- [x] **Step 4: Run Qwen3.5 executor tests and verify GREEN**

Run the Task 3 command. Expected: all tests pass with unchanged proposal and
snapshot assertions.

---

### Task 4: Focused Regression and Static Validation

**Files:**
- Modify only if a regression exposes a coordinator integration defect.

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: local evidence that generic registry, proposal KV, Qwen3.5 MTP,
  Engine publication, and gate contracts remain compatible.

- [x] **Step 1: Run focused functional suites**

Run:

```bash
PYTHONPATH=$PWD /tmp/tinyllmforge-trace-test-venv/bin/python -m pytest -q \
  tools/test_proposal_kv_lifecycle.py \
  tools/test_proposal_kv_cache.py \
  tools/test_model_runner_proposal_executor.py \
  tools/test_qwen35_mtp_executor.py \
  tools/test_qwen35_mtp_model_runner_integration.py \
  tools/test_engine_speculative_execution.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_qwen35_native_mtp_tp1_4k_engine_gate.py \
  tools/test_qwen35_native_mtp_tp4_4k_engine_gate.py
```

Expected: all tests pass.

- [x] **Step 2: Compile changed Python**

Run:

```bash
/tmp/tinyllmforge-trace-test-venv/bin/python -m py_compile \
  tinyvllm/engine/proposal_kv_lifecycle.py \
  tinyvllm/engine/qwen35_mtp_executor.py \
  tools/test_proposal_kv_lifecycle.py \
  tools/test_qwen35_mtp_executor.py
```

Expected: exit 0.

- [x] **Step 3: Check scoped whitespace**

Run:

```bash
git diff --check -- \
  tinyvllm/engine/proposal_kv_lifecycle.py \
  tinyvllm/engine/qwen35_mtp_executor.py \
  tools/test_proposal_kv_lifecycle.py \
  tools/test_qwen35_mtp_executor.py \
  docs/superpowers/specs/2026-08-14-generic-proposal-kv-lifecycle-coordinator-design.md \
  docs/superpowers/plans/2026-08-14-generic-proposal-kv-lifecycle-coordinator.md
```

Expected: exit 0.

- [x] **Step 4: Record the exact claim boundary**

Report:

```text
established:
  model-independent proposal-KV lifecycle coordinator
  Qwen3.5 native-MTP delegation with local regression coverage

not established:
  second learned model/checkpoint
  remote/GPU correctness
  performance or movement improvement
  KV8/KV4
  Phase 1 completion
```

Do not stage or commit any file.
