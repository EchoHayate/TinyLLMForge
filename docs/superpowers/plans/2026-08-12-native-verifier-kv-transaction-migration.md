# Native Verifier KV Transaction Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route the existing native single-sequence speculative verifier through `SpeculativeKVTransaction` while preserving accepted-KV direct commit and event compatibility.

**Architecture:** Keep legacy rematerialization on the old block-list API. Native mode begins a BlockManager transaction, exposes its reserved IDs to the existing proxy table, acknowledges `query_len` after target KV writes, and commits or rolls back exactly once.

**Tech Stack:** Python 3, existing dependency-light profiler tests, `BlockManager`, `SpeculativeKVTransaction`.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, or run `git clean`.
- Do not add model-name branches.
- Preserve the legacy rematerialization path.
- Preserve the existing event schema and native zero-rematerialization claim.
- Do not enable native KV offload or report simulated H2D as real savings.
- Do not claim performance improvement from this migration.
- Follow witnessed RED then minimal GREEN.

---

### Task 1: RED Native Transaction Lifecycle

**Files:**
- Modify: `tools/test_ngram_speculative.py`

**Interfaces:**
- Fake produces: `begin_speculative_kv_transaction()`.
- Fake produces: `mark_speculative_kv_materialized()`.
- Fake produces: `commit_speculative_kv_transaction()`.
- Fake produces: `rollback_speculative_kv_transaction()`.

- [x] Add transaction lifecycle recording to `_NativeBlockManager`.
- [x] Make native tests assert begin/mark/commit call arguments and terminal
  state.
- [x] Assert K=1 marks `0`.
- [x] Assert tail failure rolls back from `reserved`.
- [x] Assert commit failure rolls back from `materialized`.
- [x] Add a legacy-mode fixture assertion that transaction methods are not
  called.
- [x] Run focused native tests and confirm RED because
  `verify_and_commit_block()` still calls the old API.

### Task 2: GREEN Native Migration

**Files:**
- Modify: `tools/profile_ngram_commit.py`

**Interfaces:**
- Consumes:
  `BlockManager.begin_speculative_kv_transaction(seq, proposed_token_count)`.
- Consumes:
  `BlockManager.mark_speculative_kv_materialized(transaction, query_len)`.
- Consumes:
  `BlockManager.commit_speculative_kv_transaction(transaction, seq, accepted_tokens)`.
- Consumes:
  `BlockManager.rollback_speculative_kv_transaction(transaction, seq)`.

- [x] Branch reservation by verifier mode: transaction for native, old list
  for legacy.
- [x] Build the proxy table from the transaction's reserved IDs.
- [x] Mark native materialization after target forward and before acceptance.
- [x] Commit native accepted tokens through the transaction.
- [x] Preserve committed/released event accounting.
- [x] Roll back an active native transaction on every exception.
- [x] Preserve the original phase label and include cleanup failure details if
  rollback fails.
- [x] Run focused tests and confirm GREEN.

### Task 3: Compatibility and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run `python3 -m pytest -q tools/test_ngram_speculative.py`.
- [x] Run `python3 -m pytest -q tools/test_speculative_kv_transaction.py`.
- [x] Run the Python 3.12 `tools/test_chunked_prefill.py` script.
- [x] Run Python 3.9 and Python 3.12 `py_compile` for changed Python files.
- [x] Run `git diff --check` and verify staged diff is empty.
- [x] Record exact evidence and limitations in the handoff.
- [x] Mark checkboxes only from fresh evidence.

## Fresh Completion Evidence

```text
focused native lifecycle:
7 passed, 52 deselected

python3 -m pytest -q tools/test_ngram_speculative.py
59 passed

python3 -m pytest -q tools/test_speculative_kv_transaction.py
25 passed

PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.12 \
  tools/test_chunked_prefill.py
chunked prefill tests passed

Python 3.9 py_compile:  PASS
Python 3.12 py_compile: PASS
git diff --check:       PASS
staged diff:            empty
```

The legacy tests use block-manager fixtures without the new transaction API;
their inclusion in the 59-test green matrix proves the legacy path remains
on the old API.
