# Explicit Prefill Publication Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the explicit Scheduler hook-to-source-publisher-to-publication-transaction lifecycle with real dependency-light components.

**Architecture:** A one-rank fake Engine transport hosts the real Scheduler, source publisher, and Engine transaction coordinator. Publication phase stubs assert source liveness and return schema-valid acknowledgements.

**Tech Stack:** Python, existing Scheduler/BlockManager/hybrid allocator/candidate/source publisher/Engine coordinator, plain-assert test runner.

## Global Constraints

- Adaptive-ngram worktree only.
- Integration test only; no new automatic production wiring.
- No subagents, staging, commits, or evidence cleanup.
- Do not claim real-model, CUDA memory, quality, or speed benefit.

---

### Task 1: Success and Default-Off Integration

- [x] Add a real-component one-rank fixture.
- [x] Prove default-off prefill performs zero publication phases.
- [x] Prove explicit install runs prepare/precommit/finalize/seal once.
- [x] Assert each phase runs before append/release with exact live identity.
- [x] Confirm focused tests GREEN.

### Task 2: Failure Ordering

- [x] Inject a publication phase error.
- [x] Assert no completion token append and no storage release.
- [x] Assert future Scheduler calls fail closed.
- [x] Confirm focused tests GREEN.

### Task 3: Regression and Handoff

- [x] Run hook, installation, source publisher, transaction, restore, and
  chunked-prefill suites.
- [x] Run compile, diff-check, staged-file, and automatic wiring audits.
- [x] Record integration proof and remaining real-runtime boundary.
