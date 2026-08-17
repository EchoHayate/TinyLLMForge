# Identity-Driven Source Publisher Installation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a zero-argument explicit publisher installer driven only by the Engine canonical runtime identity.

**Architecture:** The new method validates the stored identity and delegates to the existing source-publisher installer. It introduces no new publication state or automatic runtime call.

**Tech Stack:** Python, existing Engine identity and source-publisher installer, dependency-light AST method tests.

## Global Constraints

- Adaptive-ngram worktree only.
- No duplicate caller-supplied identity fields.
- No automatic call from `LLMEngine.__init__()` or `step()`.
- No subagents, staging, commits, or evidence cleanup.
- Do not claim production memory or speed benefit.

---

### Task 1: RED/GREEN Canonical Delegation

- [x] Add missing-identity and exact delegation tests.
- [x] Observe missing method RED.
- [x] Implement minimal zero-argument delegate.
- [x] Confirm focused tests GREEN.

### Task 2: Idempotency and Conflict

- [x] Add exact-repeat idempotency test.
- [x] Add conflicting manual installation rejection test.
- [x] Assert `LLMEngine.step()` remains disconnected.
- [x] Confirm focused tests GREEN.

### Task 3: Regression and Handoff

- [x] Run identity, installation, integration, publication, and restore suites.
- [x] Run compile, diff-check, staged-file, and automatic wiring audits.
- [x] Record canonical-installer proof and checkpoint-worker gap.
