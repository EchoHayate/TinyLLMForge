# Qwen3.5 Source-Bound Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit synchronous capture-revalidate-publish path for one live aligned Qwen3.5 source request.

**Architecture:** Candidate revalidation recaptures exact source identity. A single-thread source publisher owns monotonic ticket IDs and invokes the already-proven Engine publication transaction only after source equality succeeds.

**Tech Stack:** Python 3.9, frozen dataclasses, existing candidate and Engine transaction APIs, dependency-light CPU tests.

## Global Constraints

- Adaptive-ngram worktree only.
- Same Engine control thread; no concurrent Scheduler mutation.
- No `LLMEngine.step()`, Scheduler postprocess, or ModelRunner `run()` wiring.
- No subagents, staging, commits, or evidence cleanup.
- Do not claim production memory or speed benefit.

---

### Task 1: RED/GREEN Candidate Revalidation

- [x] Add exact source revalidation and drift rejection tests.
- [x] Observe the missing method RED.
- [x] Implement equality-based recapture validation.
- [x] Confirm candidate tests GREEN.

### Task 2: RED/GREEN Source Publisher

- [x] Add success, false propagation, monotonic ticket, missing-owner,
  uninstalled-coordinator, and reentrancy tests.
- [x] Observe the missing module RED.
- [x] Implement the minimal synchronous publisher.
- [x] Confirm focused tests GREEN.

### Task 3: Regression and Handoff

- [x] Run candidate, publication transaction/transport/participant/cache, and
  restore suites.
- [x] Run compile, diff-check, staged-file, and runtime source audits.
- [x] Record single-thread-only and no-lifetime-pin boundaries.
