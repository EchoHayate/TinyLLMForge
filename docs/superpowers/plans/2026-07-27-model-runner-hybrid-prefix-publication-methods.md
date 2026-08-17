# ModelRunner Hybrid Prefix Publication Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install the publication participant in the shared owner graph and expose five validated ModelRunner command methods.

**Architecture:** Restore owner builds both participants over one snapshot cache. ModelRunner keeps independent restore/publication participant references and returns strict acknowledgement dictionaries for publication commands.

**Tech Stack:** Python 3.9, dataclasses, AST dependency-light tests, existing participant/cache.

## Global Constraints

- Adaptive-ngram worktree only.
- No Engine transport or runtime `step()` wiring.
- No subagents, staging, commits, or evidence cleanup.
- Preserve restore behavior and exact cache semantics.
- Do not claim production benefit.

---

### Task 1: RED Owner and Installation

- [x] Extend owner factory tests for shared publication participant coherence.
- [x] Add ModelRunner install type/rank/pool/idempotency tests.
- [x] Observe missing fields/methods RED.

### Task 2: GREEN Owner and Installation

- [x] Add owner publication participant.
- [x] Add ModelRunner state and install method.
- [x] Configure both participants from one owner.
- [x] Confirm owner/install GREEN.

### Task 3: RED/GREEN Publication Methods

- [x] Test prepare/precommit/finalize/seal/rollback delegation.
- [x] Test exact allowed statuses and malformed acknowledgement rejection.
- [x] Test uninstalled fail-closed behavior.
- [x] Implement strict result helper and five methods.

### Task 4: Regression and Handoff

- [x] Run owner, restore/publication methods, publication stack, and adjacent
  suites.
- [x] Compile, diff-check, staged-file check.
- [x] Record transport-disconnected claim boundary.
