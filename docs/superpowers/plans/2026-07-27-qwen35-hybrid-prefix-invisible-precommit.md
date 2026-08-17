# Qwen3.5 Hybrid Prefix Invisible Precommit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete all fallible exact tensor interning before a snapshot becomes visible.

**Architecture:** Extend the single in-flight staged state with a precommitted immutable snapshot and acquired intern refs. Precommit is invisible and reversible; finalize performs only entry/accounting mutation; immediate commit composes both phases.

**Tech Stack:** Python 3.9, PyTorch CPU, existing staged cache and exact interning.

## Global Constraints

- Adaptive-ngram worktree only.
- No Engine/Scheduler/ModelRunner/transport wiring.
- No subagents, staging, commits, or evidence cleanup.
- Preserve exact bytes and all previous tests.
- Do not claim all-rank or production benefit.

---

### Task 1: RED Invisible Precommit

- [x] Add tests proving precommit leaves entries and visible bytes unchanged.
- [x] Assert reserved intern refs/bytes are separately observable.
- [x] Observe missing API RED.

### Task 2: GREEN Precommit and Rollback

- [x] Extend staged state with prepared/precommitted phases.
- [x] Acquire intern refs and build snapshot in precommit.
- [x] Roll back all refs/counters on failure.
- [x] Rollback a precommitted transaction exactly.

### Task 3: RED/GREEN Non-Fallible Finalize

- [x] Count digest/equality/intern acquisition calls during finalize and require
  zero.
- [x] Test exact replacement, LRU, and handle consumption.
- [x] Implement finalize using only prebuilt snapshot/ref state.
- [x] Make commit compose precommit plus finalize.

### Task 4: Regression and Handoff

- [x] Run cache, participant, and adjacent suites.
- [x] Compile, diff-check, and staged-file check.
- [x] Record private reserved versus visible bytes and remaining all-rank gap.
