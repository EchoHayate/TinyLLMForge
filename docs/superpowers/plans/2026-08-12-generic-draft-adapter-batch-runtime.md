# Generic Draft Adapter and Batch Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a model-independent draft proposal adapter contract and a
dependency-light batch-native speculative orchestration core with per-sequence
transactional KV commit/rollback.

**Architecture:** `adapter.py` validates immutable proposal capabilities,
contexts, and results. `batch_runtime.py` executes one first-target callback,
one adapter batch, at most one tail-verifier callback, and independent
per-sequence transactions while preserving stable input ordering and explicit
partial-commit failures.

**Tech Stack:** Python 3.9+, dataclasses, typing Protocol, existing
`SpecVerifyPlan` and `BlockManager` speculative KV transaction APIs, pytest.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, or run
  `git clean`.
- No model-name or source-name behavior branches in runtime code.
- Accepted KV commits directly; rejected suffixes roll back without token
  replay, KV copy, or rematerialization.
- Keep this slice dependency-light and CPU-testable.
- Do not modify scheduler, engine, model-runner, CUDA Graph, TP collective,
  KV-offload, KV4/KV8, or Qwen3.5-specific behavior.
- No performance claim.

---

### Task 1: Draft Adapter Contract

**Files:**
- Create: `tinyvllm/speculative/adapter.py`
- Create: `tools/test_speculative_adapter.py`

**Interfaces:**
- Consumes: Python dataclasses and `typing.Protocol`.
- Produces:
  - `DraftCapabilities`
  - `DraftContext`
  - `DraftProposal`
  - `DraftAdapter`
  - `validate_draft_adapter_batch(adapter, contexts)`

- [x] **Step 1: Write RED tests for valid and empty proposals**

Create fixtures with two context rows, one non-empty proposal, and one empty
proposal. Assert stable input ordering and immutable tuple output:

```python
proposals = validate_draft_adapter_batch(adapter, contexts)
assert proposals[0].token_ids == (10, 11)
assert proposals[1].token_ids == ()
```

- [x] **Step 2: Write RED tests for fail-closed validation**

Cover duplicate/missing/extra sequence IDs, boolean tokens, capability/context
length overflow, proposal-over-remaining-budget acceptance, missing required
hidden/logits, negative/non-finite timing, invalid `source_type`, and mutation
of a mutable sequence-history fixture after context construction.

- [x] **Step 3: Run RED**

Run:

```bash
python3 -m pytest -q tools/test_speculative_adapter.py
```

Expected: collection failure because
`tinyvllm.speculative.adapter` does not exist.

- [x] **Step 4: Implement immutable types and validation**

Implement:

```python
def validate_draft_adapter_batch(
    adapter: DraftAdapter,
    contexts: tuple[DraftContext, ...],
) -> tuple[DraftProposal, ...]:
    ...
```

Validation must normalize no values silently. It verifies exact unique ID
coverage, required payload capabilities, integer non-boolean tokens, proposal
limits, matching `source_type`, and finite non-negative timing values. Return
proposals in context order regardless of adapter result order.

- [x] **Step 5: Run GREEN**

Run:

```bash
python3 -m pytest -q tools/test_speculative_adapter.py
```

Expected: all adapter tests pass.

---

### Task 2: Batch-Native Transaction Runtime

**Files:**
- Create: `tinyvllm/speculative/batch_runtime.py`
- Create: `tools/test_speculative_batch_runtime.py`

**Interfaces:**
- Consumes:
  - `DraftContext`, `DraftProposal`, `validate_draft_adapter_batch`
  - `SpecVerifyPlan`, `build_spec_verify_plan`
  - `BlockManager.begin_speculative_kv_transaction`
  - `BlockManager.mark_speculative_kv_materialized`
  - `BlockManager.commit_speculative_kv_transaction`
  - `BlockManager.rollback_speculative_kv_transaction`
- Produces:
  - `FirstTargetResult`
  - `TailBatchItem`
  - `TailBatchResult`
  - `NativeSpeculativeSequenceResult`
  - `NativeSpeculativeBatchResult`
  - `NativeSpeculativeBatchError`
  - `execute_native_speculative_batch(...)`

- [x] **Step 1: Write RED tests for callback batching**

Use four fake sequences with K=0, K=1, K=2, and K=4 proposals. Assert:

```python
assert first_target_callback_count == 1
assert tail_callback_count == 1
assert tail_items_sequence_ids == (k2_id, k4_id)
```

Return first-target and tail rows out of order and assert results remain in
input sequence order.

- [x] **Step 2: Write RED tests for acceptance and transaction lifecycle**

Cover zero, one, partial, and full greedy acceptance, EOS truncation, output
budget truncation, K=1 materialized count zero, direct accepted block commit,
rejected block release, and empty-proposal no-transaction behavior.

- [x] **Step 3: Write RED tests for failures**

Inject failures in first-target callback, adapter proposal, second reservation,
tail callback, tail-result validation, second materialization, acceptance, and
second commit. Assert:

- every uncommitted transaction rolls back exactly once;
- committed sequence IDs are reported for partial commit;
- rollback failures are attached without replacing the original cause;
- no tail callback runs when every proposal is empty or K=1.

- [x] **Step 4: Run RED**

Run:

```bash
python3 -m pytest -q tools/test_speculative_batch_runtime.py
```

Expected: collection failure because
`tinyvllm.speculative.batch_runtime` does not exist.

- [x] **Step 5: Implement batch types, validation, and orchestration**

Implement:

```python
def execute_native_speculative_batch(
    *,
    block_manager,
    seqs: tuple[object, ...],
    draft_adapter: DraftAdapter,
    eos_token: int,
    run_first_targets: Callable[
        [tuple[object, ...]],
        tuple[FirstTargetResult, ...],
    ],
    run_tail_batch: Callable[
        [tuple[TailBatchItem, ...]],
        tuple[TailBatchResult, ...],
    ],
) -> NativeSpeculativeBatchResult:
    ...
```

Requirements:

1. validate unique sequence IDs;
2. call first-target exactly once;
3. build adapter contexts from immutable token snapshots;
4. begin one transaction for each non-empty proposal;
5. call tail exactly once only when at least one plan has `query_len > 0`;
6. validate all tail rows before materialization;
7. materialize all active transactions before acceptance/commit;
8. compute per-row exact greedy prefix, EOS, and budget truncation;
9. commit in stable input order;
10. rollback every active uncommitted transaction on error;
11. return stable ordered sequence results and non-negative phase timings.

- [x] **Step 6: Run GREEN**

Run:

```bash
python3 -m pytest -q tools/test_speculative_batch_runtime.py
```

Expected: all batch runtime tests pass.

---

### Task 3: Public Exports, Regression, and Handoff

**Files:**
- Modify: `tinyvllm/speculative/__init__.py`
- Modify: `docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: completed adapter and batch runtime public types.
- Produces: package exports and exact validation/limitation evidence.

- [x] **Step 1: Add explicit package exports**

Export adapter and batch-runtime public contracts without importing model,
engine, scheduler, CUDA, or profiler modules.

- [x] **Step 2: Run focused and regression matrices**

Run:

```bash
python3 -m pytest -q \
  tools/test_speculative_adapter.py \
  tools/test_speculative_batch_runtime.py \
  tools/test_speculative_runtime.py \
  tools/test_speculative_kv_transaction.py \
  tools/test_ngram_speculative.py

PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.12 \
  tools/test_chunked_prefill.py
```

Expected: all focused pytest tests pass and chunked prefill prints
`chunked prefill tests passed`.

- [x] **Step 3: Run compatibility checks**

Run Python 3.9 and Python 3.12 `py_compile` on the two new modules and tests,
then run:

```bash
git diff --check
test -z "$(git diff --cached --name-only)"
```

Expected: compile succeeds, diff check succeeds, and staged diff is empty.

- [x] **Step 4: Update audit and handoff**

Record:

- adapter and batch core evidence;
- exact test counts;
- direct accepted-KV and rollback guarantees;
- no scheduler/model-runner/MTP/GPU/TP/performance claim;
- next gate: scheduler and model-runner integration using the new callback
  boundary.

## Completion Evidence

```text
adapter contract:
  28 passed
batch runtime:
  15 passed
public API:
  1 passed
combined adapter/batch/public/single-runtime/transaction/profiler:
  142 passed
chunked prefill:
  passed
Python 3.9 and 3.12 py_compile:
  passed
generic source scan:
  passed
git diff --check:
  passed
staged diff:
  empty
```

The scoped dependency-light adapter and batch runtime are complete.
Production scheduler/model-runner integration, concrete n-gram/SAM adapters,
learned drafter/MTP execution, GPU/TP validation, and performance promotion
remain subsequent gates.
