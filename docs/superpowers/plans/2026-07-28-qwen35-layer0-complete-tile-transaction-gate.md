# Qwen3.5 Layer-0 Complete Tile Transaction Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream, independently verify, copy, completely cover, isolate, and roll back all 14 real checkpoint bindings targeting layer 0 for TP=1 and TP=2.

**Architecture:** Reuse the five-transform source-bound worker and CPU target. Derive every layer-0 production tile under a fixed 64 KiB budget, stream each tile through two shard descriptors, aggregate per-binding evidence, validate the intentional gate/up shared destination, then restore 13 unique destination snapshots in reverse order.

**Tech Stack:** Python standard library, `os.pread`, SHA256, PyTorch CPU BF16/F32 tensors, existing Qwen3.5 tile planner/copy primitive, source-bound SSH orchestration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Use only `sitian@10.232.195.203`.
- Select exactly binding indices `1..14`.
- Use `max_tile_bytes=65536`.
- Preserve the exact aggregate counts in the written design.
- Open the shard exactly twice per rank.
- Stream at most one production tile, one verifier tile, and one decoded tile.
- Never call a checkpoint loader, assignment, `target.take()`, candidate
  installation, forward, CUDA, Engine, publication, or restore.
- Preserve failed/superseded evidence and schema-v2 canonical `NO_GO`.
- Do not stage, commit, merge, overwrite, or delete evidence.

---

### Task 1: Layer Contract and Generic Range Derivation TDD

**Files:**
- Create: `tools/test_qwen35_real_checkpoint_layer0_transaction_preflight.py`
- Create: `tools/qwen35_real_checkpoint_layer0_transaction_preflight.py`

- [x] Write failing tests for the 40-file closure, binding indices `1..14`,
  unique destination count `13`, alias group `[12,13]`, aggregate tile/kind/
  byte/range contracts, and memory ceilings.
- [x] Run RED and confirm the module is absent.
- [x] Implement generic source-range derivation for rank-1, full-row rank-2,
  TP axis-1 row spans, and squeezed convolution tiles.
- [x] Add synthetic tests that prove exact range arithmetic and reject
  out-of-metadata, unsorted, overlapping, shape, dtype-width, and byte-count
  mismatches.
- [x] Run focused GREEN and `py_compile`.

### Task 2: Streaming Layer Transaction TDD

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_layer0_transaction_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_layer0_transaction_preflight.py`

- [x] Write failing synthetic tests for two open descriptors, deterministic
  tile order, one-tile-at-a-time lifetime, per-binding rolling hashes,
  complete destination coverage, shared destination composition, non-layer
  isolation, and reverse unique-object rollback.
- [x] Require fail-closed behavior for missing/duplicate/out-of-order tiles,
  duplicate non-shared aliases, gate/up overlap or gap, incomplete binding
  bytes, non-layer mutation, short read, hash mismatch, and rollback failure.
- [x] Implement streaming production/verifier reads, exact decode/copy,
  per-binding aggregates, layer coverage validation, and reverse rollback.
- [x] Run focused GREEN.

### Task 3: Fresh-Process Worker and Atomic Publication TDD

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_layer0_transaction_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_layer0_transaction_preflight.py`

- [x] Write failing tests for deterministic 40-file staging, three fresh rank
  processes, no-CUDA/thread environment, separate finalizer, source binding,
  partial-failure non-publication, and atomic two-artifact output.
- [x] Implement row/aggregate validators, CLI modes `run`,
  `internal-rank-worker`, `internal-finalize`, and `validate`.
- [x] Run focused GREEN, compile, exact closure, forbidden-call AST,
  two-descriptor streaming AST, worker rejection, `git diff --check`, and
  staged-file audits.

### Task 4: Live Layer-0 Gate and Handoff

**Files:**
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/layer0_transaction_preflight.json`
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/specs/2026-07-28-qwen35-layer0-complete-tile-transaction-gate-design.md`
- Modify:
  `docs/superpowers/plans/2026-07-28-qwen35-layer0-complete-tile-transaction-gate.md`

- [x] Run one unique source-bound remote gate with three independent rank
  processes.
- [x] Independently verify source ranges and per-binding hashes, TP slicing,
  replicated equality, axis-1 reconstruction, segmented offsets, complete
  layer bytes, shared gate/up destination composition, isolation, and rollback.
- [x] Verify exact remote inventory, source/artifact hashes, PIDs, memory,
  counters, and no partial evidence.
- [x] Run layer0, bundle, one-tile, CPU, meta, loader, metadata, reader,
  worker, factory, binding, authorization, and safety regressions.
- [x] Append exact evidence and the next gate boundary to the handoff; mark
  checkboxes only after fresh verification.

## Completed Run

```text
run tag:
  qwen35-layer0-transaction-20260728-053410
status:
  PASS
source tree SHA256:
  0be5c56dd5c49f4e257d14fbb478e5ae6170a0b529852c537272897b23d679fd
record SHA256:
  167d3ee5e3b0996ebab9331f17e36d1775ea775c39964d2f4c1e4de3c9820b73
manifest SHA256:
  cb3e7e4e79590eb77c3e13a74b0afc38f89e08b9ded98bbe8cdbc1e54c7aee93
fresh PIDs:
  2959409, 2961371, 2962718
independent direct-pread checks:
  62 passed
```

Fresh regression matrix:

```text
layer0 transaction preflight: passed (6 tests)
five-transform bundle preflight: passed (6 tests)
one-tile payload preflight: passed (6 tests)
CPU materialization preflight: passed (6 tests)
meta target preparation preflight: passed (7 tests)
loader construction preflight: passed (5 tests)
metadata-header preflight: passed (6 tests)
bounded metadata reader: passed (4 tests)
worker helper: passed (6 tests)
candidate factory: passed (6 tests)
real component binding: passed (1 test)
loader configuration: passed (4 tests)
candidate loader: passed (5 tests)
authorization: passed
safety gate: passed (23 tests)
focused py_compile: passed
frozen 40-file source closure: passed
forbidden-call AST scan: passed
exact two-descriptor streaming AST scan: passed
worker direct execution rejection: passed
git diff --check: passed
staged files: 0
```
