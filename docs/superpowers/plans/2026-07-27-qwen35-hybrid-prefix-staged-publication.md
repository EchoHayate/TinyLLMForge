# Qwen3.5 Hybrid Prefix Staged Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate exact snapshot capture from visible cache publication with a bounded one-shot prepare/commit/rollback API.

**Architecture:** Keep one private prepared snapshot per cache. Prepare clones and validates state without interning or changing visible entries; commit interns and publishes atomically; rollback discards private clones. Existing `publish()` composes the new transaction locally.

**Tech Stack:** Python 3.9, PyTorch CPU, dataclasses, existing exact tensor interning and cross-layer transaction.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not stage, commit, merge, or delete experiment evidence.
- Do not modify Engine, Scheduler, ModelRunner, transport, or live publication wiring.
- Preserve exact byte-identical tensor interning and restore semantics.
- Permit one prepared publication per cache.
- Do not claim production cache or speed improvement.

---

### Task 1: RED Prepare Visibility and Rollback

**Files:**
- Modify: `tools/test_qwen35_hybrid_prefix_cache.py`

- [x] Add tests for prepare leaving visible entries/bytes/LRU unchanged.
- [x] Mutate source state after prepare and prove later commit uses staged
  clones.
- [x] Prepare a same-key replacement, rollback it, and prove the old entry
  remains exact and restorable.
- [x] Run the focused suite and observe missing staged-publication API failure.

### Task 2: GREEN Single In-Flight Staging

**Files:**
- Modify: `tinyvllm/engine/qwen35_hybrid_prefix_cache.py`

- [x] Add the frozen public handle and private staged-state record.
- [x] Extract clone/validation from `publish()` into
  `prepare_publication()`.
- [x] Enforce one cache-bound in-flight handle.
- [x] Add prepared-byte and lifecycle observations.
- [x] Implement `rollback_publication()` and confirm Task 1 GREEN.

### Task 3: RED/GREEN Commit and Handle Safety

**Files:**
- Modify: `tools/test_qwen35_hybrid_prefix_cache.py`
- Modify: `tinyvllm/engine/qwen35_hybrid_prefix_cache.py`

- [x] Test successful commit publishes exact staged values and consumes the
  handle.
- [x] Test foreign, stale, committed, rolled-back, and replayed handles reject
  before mutation.
- [x] Test a second prepare conflict.
- [x] Test oversize prepare returns `None`.
- [x] Inject commit interning failure and prove previous entry/counters remain
  unchanged while explicit rollback remains possible.
- [x] Implement `commit_publication()` and exact handle validation.

### Task 4: Preserve Immediate Publish

**Files:**
- Modify: `tinyvllm/engine/qwen35_hybrid_prefix_cache.py`
- Modify: `tools/test_qwen35_hybrid_prefix_cache.py`

- [x] Refactor `publish()` to prepare then commit.
- [x] On commit exception, rollback the still-live prepared handle.
- [x] Verify all existing exact interning, collision, LRU, invalidation,
  restore, FP32, and BF16 tests remain green.

### Task 5: Verification and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run focused and eight adjacent prefix/state suites.
- [x] Run Python compile, `git diff --check`, and staged-file check.
- [x] Record staging bytes, handle lifecycle, failure atomicity, and claim
  boundary.
- [x] Update the long-term objective audit and identify the all-rank
  participant/coordinator as the next gate.
