# Qwen3.5 Runtime Prefix Identity Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind and all-rank validate one canonical model/layout/dtype identity for Qwen3.5 prefix publication.

**Architecture:** A focused immutable identity type validates manifest SHA256 and derives layout/dtype from the bound model owner. ModelRunner stores it once; Engine aggregates pickle-safe rank rows without enabling publication automatically.

**Tech Stack:** Python, torch dtype enums, existing Qwen3.5 model owner and acknowledged ModelRunner command channel, dependency-light plain-assert tests.

## Global Constraints

- Adaptive-ngram worktree only.
- Model fingerprint must be lowercase SHA256, never a path or arbitrary label.
- Layout and dtype must be owner-derived.
- No automatic publisher installation or `LLMEngine.step()` changes.
- No subagents, staging, commits, or evidence cleanup.
- Do not claim production memory or speed benefit.

---

### Task 1: RED/GREEN Canonical Identity

- [x] Add SHA256, owner, layout, and dtype derivation tests.
- [x] Add mixed-dtype rejection test.
- [x] Observe missing identity module RED.
- [x] Implement immutable identity and rank-row serialization.
- [x] Confirm focused tests GREEN.

### Task 2: RED/GREEN ModelRunner Binding

- [x] Add first-bind and exact-repeat tests.
- [x] Add replacement/owner-drift rejection tests.
- [x] Implement single-assignment ModelRunner method and state.
- [x] Confirm focused tests GREEN.

### Task 3: RED/GREEN Engine All-Rank Aggregation

- [x] Add complete all-rank aggregation test.
- [x] Add rank/model/layout/dtype mismatch poison tests.
- [x] Add repeated configuration and replacement tests.
- [x] Implement acknowledged Engine configuration method.
- [x] Assert `LLMEngine.step()` remains identity/publication-free.

### Task 4: Regression and Handoff

- [x] Run owner, restore, publication, Scheduler integration, and command-ack
  suites.
- [x] Run compile, diff-check, staged-file, and automatic wiring audits.
- [x] Record canonical identity proof and manifest-worker gap.
