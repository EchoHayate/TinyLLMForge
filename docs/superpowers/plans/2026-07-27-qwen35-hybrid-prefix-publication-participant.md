# Qwen3.5 Hybrid Prefix Publication Participant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add rank-local ticket-bound prepare/commit/rollback semantics over cache-local staged publication.

**Architecture:** A validated immutable payload binds ticket, request, exact prefix identity, block identity, and lease. The participant tracks prepared and terminal tickets, delegates snapshot ownership to the cache, and provides idempotent acknowledgements for later all-rank coordination.

**Tech Stack:** Python 3.9, dataclasses, existing hybrid state lease and staged snapshot cache.

## Global Constraints

- Modify only the adaptive-ngram worktree.
- No Engine, Scheduler, ModelRunner, or transport wiring.
- No subagents, staging, commits, or experiment cleanup.
- Preserve exact staged publication and interning semantics.
- Do not claim all-rank atomicity or production benefit.

---

### Task 1: RED Payload and Prepare

- [x] Create focused participant tests with a real pool/cache fixture.
- [x] Test payload validation, invisible prepare, idempotent prepare, changed
  payload rejection, and oversize rejection.
- [x] Observe missing-module RED.

### Task 2: GREEN Participant State Machine

- [x] Add payload and acknowledgement dataclasses.
- [x] Add participant constructor coherence validation.
- [x] Implement prepared and terminal ticket maps.
- [x] Implement prepare and confirm GREEN.

### Task 3: RED/GREEN Commit and Rollback

- [x] Test exact commit, idempotent commit, exact rollback, and idempotent
  rollback.
- [x] Test committed-to-rollback and rolled-back-to-commit rejection.
- [x] Inject cache commit failure and prove the ticket remains prepared for
  rollback.
- [x] Implement commit/rollback and confirm focused GREEN.

### Task 4: Regression and Handoff

- [x] Run participant, cache, acquisition, restore, transaction, owner,
  Engine, and ModelRunner adjacent suites.
- [x] Compile, run diff check, and confirm zero staged files.
- [x] Record proof and keep the all-rank coordinator as the next gate.
