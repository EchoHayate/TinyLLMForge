# Qwen3.5 Hybrid Prefix Reversible Finalize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow finalized hybrid-prefix snapshots to be rolled back until an explicit seal commits every rank.

**Architecture:** Journal replaced and evicted visible entries without releasing their intern ownership. Finalize mutates visibility reversibly, rollback restores exact LRU/accounting, and seal releases journal ownership. Coordinator finalizes every rank, rolls back all on finalize failure, then seals all.

**Tech Stack:** Python 3.9, OrderedDict, existing intern refcounts and publication participant/coordinator.

## Global Constraints

- Adaptive-ngram worktree only.
- No Engine/Scheduler/ModelRunner/runtime wiring.
- No subagents, staging, commits, or evidence cleanup.
- Exact entry values, LRU, refs, and counters must restore.
- Do not claim crash-consensus or production benefit.

---

### Task 1: RED Cache Finalize Rollback

- [x] Test new-entry finalize then rollback.
- [x] Test replacement finalize then rollback restores old snapshot and LRU.
- [x] Test byte-limit evictions restore exactly.
- [x] Observe current finalized handle cannot roll back.

### Task 2: GREEN Journal and Seal

- [x] Add reversible finalize journal.
- [x] Defer release of replaced/evicted snapshot refs until seal.
- [x] Implement rollback from finalized state.
- [x] Implement seal and local commit composition.

### Task 3: Participant and Coordinator Recovery

- [x] Add participant finalize/seal operations and terminal semantics.
- [x] Change coordinator to finalize all, rollback all finalized/precommitted
  ranks on failure, then seal all.
- [x] Test injected rank-1 finalize failure restores rank-0 invisibility.
- [x] Poison on seal failure.

### Task 4: Regression and Handoff

- [x] Run publication/cache and adjacent suites.
- [x] Compile, diff-check, staged-file check.
- [x] Record injected partial-finalize recovery and remaining crash boundary.
