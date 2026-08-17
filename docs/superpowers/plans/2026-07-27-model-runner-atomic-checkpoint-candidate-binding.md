# ModelRunner Atomic Checkpoint Candidate Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Atomically bind a loaded checkpoint candidate's exact owner and canonical runtime identity to ModelRunner.

**Architecture:** Prederive identity before mutation, reuse the existing owner binder's fail-before-assignment checks, then publish already-validated identity fields. Exact complete repeats are idempotent; partial or conflicting state fails closed.

**Tech Stack:** Python, existing loaded candidate, model owner binder, runtime identity helper, dependency-light AST method tests.

## Global Constraints

- Adaptive-ngram worktree only.
- Exact streaming candidate type only.
- No partial-state repair or implicit provenance upgrade.
- No Engine orchestration or automatic runtime enablement.
- No subagents, staging, commits, or evidence cleanup.
- Do not claim production memory or speed benefit.

---

### Task 1: RED/GREEN First Atomic Bind

- [x] Add exact candidate first-bind test.
- [x] Assert owner, bridge, identity, and owner identity pointer.
- [x] Observe missing method RED.
- [x] Implement prevalidated owner+identity binding.
- [x] Confirm focused tests GREEN.

### Task 2: RED/GREEN Idempotency and Failure Atomicity

- [x] Add exact-repeat test.
- [x] Add type/fingerprint/model/owner-graph rejection tests.
- [x] Add partial-state and different-candidate rejection tests.
- [x] Assert every failure leaves state unchanged.
- [x] Confirm focused tests GREEN.

### Task 3: Regression and Handoff

- [x] Run loading, publication slot, owner, identity, and command-ack suites.
- [x] Run compile, diff-check, staged-file, and automatic wiring audits.
- [x] Record atomic binding proof and Engine orchestration gap.
