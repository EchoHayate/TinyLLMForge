# Qwen3.5 Hybrid Prefix Publication Coordinator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Coordinate exact rank-local staged publication with rollback through every non-visible phase and fail-stop handling for unexpected finalize failures.

**Architecture:** Validate a complete rank/payload matrix, run prepare and precommit barriers, rollback all participants on any pre-visible failure, then finalize all participants. Poison on rollback or finalize inconsistency rather than overclaiming recoverable strong atomicity.

**Tech Stack:** Python 3.9, existing publication payload/participant/cache, CPU dependency-light tests.

## Global Constraints

- Adaptive-ngram worktree only.
- No Engine, Scheduler, ModelRunner, or transport wiring.
- No subagents, staging, commits, or evidence cleanup.
- Do not claim recoverable partial-finalize atomicity.
- Preserve exact cache semantics and prior tests.

---

### Task 1: RED Success and Matrix Validation

- [x] Build two independent rank-local pools/caches/participants.
- [x] Test successful two-rank prepare/precommit/finalize.
- [x] Test missing, duplicate, wrong TP, and cross-rank identity mismatch.
- [x] Observe missing coordinator module RED.

### Task 2: GREEN Coordinator Core

- [x] Validate participant IDs and payload matrix.
- [x] Validate acknowledgements exactly.
- [x] Implement success path and focused GREEN.

### Task 3: RED/GREEN Rollback and Poisoning

- [x] Inject prepare rejection/error and assert earlier ranks roll back.
- [x] Inject precommit failure and assert every cache remains invisible.
- [x] Inject rollback failure and assert poison.
- [x] Inject finalize failure after one rank and assert poison plus reuse block.
- [x] Implement rollback orchestration and fail-stop poisoning.

### Task 4: Regression and Handoff

- [x] Run coordinator, participant, cache, and adjacent suites.
- [x] Compile, diff-check, and staged-file check.
- [x] Record proven rollback boundary and reversible-finalize requirement.
