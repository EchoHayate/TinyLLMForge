# Scheduler Prefill Commit Hook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-off one-shot Scheduler callback at the complete-prefill metadata-stable, resource-live boundary.

**Architecture:** One helper runs after commit/update and before sample/release across legacy, chunked, and mixed paths. Successful request IDs are deduplicated; callback failure poisons future scheduling.

**Tech Stack:** Python 3.9, existing Scheduler/Sequence/BlockManager, dependency-light CPU tests.

## Global Constraints

- Adaptive-ngram worktree only.
- No Engine publisher installation or `LLMEngine.step()` publication call.
- No subagents, staging, commits, or evidence cleanup.
- Hook remains default-off and synchronous.
- Do not claim production memory or speed benefit.

---

### Task 1: RED/GREEN Hook Installation and Legacy Ordering

- [x] Add default-off, callable/type, idempotency, replacement tests.
- [x] Add legacy prefill ordering/resource-liveness test.
- [x] Observe missing method/helper RED.
- [x] Implement minimal hook state, installation, and helper.
- [x] Confirm focused tests GREEN.

### Task 2: RED/GREEN Chunked and Mixed Coverage

- [x] Add non-final chunk suppression and final chunk one-shot tests.
- [x] Add mixed prefill ordering test.
- [x] Implement helper calls in both paths.
- [x] Confirm focused and chunked-prefill regression suites GREEN.

### Task 3: RED/GREEN Fail-Stop

- [x] Add hook-exception test proving no token append/release.
- [x] Add future schedule poison test.
- [x] Implement poison state and schedule guard.
- [x] Confirm focused tests GREEN.

### Task 4: Regression and Handoff

- [x] Run Scheduler, chunked prefill, publication, transaction, restore, and
  hybrid-state suites.
- [x] Run compile, diff-check, staged-file, and runtime wiring audits.
- [x] Record callback-boundary proof and publisher-disconnected boundary.
