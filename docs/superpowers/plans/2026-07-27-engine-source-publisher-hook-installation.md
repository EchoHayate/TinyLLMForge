# Engine Source Publisher Hook Installation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit default-off Engine method that installs the source-bound publisher at the Scheduler prefill-commit hook.

**Architecture:** Engine owns one publisher and one stable bound hook callable. Exact repeated configuration is idempotent; replacement and partial installation fail closed. `LLMEngine.__init__()` and `step()` remain publication-inactive.

**Tech Stack:** Python, existing `LLMEngine`, `Scheduler`, and `Qwen35HybridPrefixSourcePublisher`, dependency-light AST method tests.

## Global Constraints

- Adaptive-ngram worktree only.
- No automatic installation from `LLMEngine.__init__()` or `LLMEngine.step()`.
- No model fingerprint inference.
- No subagents, staging, commits, or evidence cleanup.
- Do not claim production memory or speed benefit.

---

### Task 1: RED/GREEN Explicit Installation

- [x] Add default-off and first-install tests.
- [x] Add exact configuration and stable callback assertions.
- [x] Observe missing import/method RED.
- [x] Implement minimal Engine import, attributes, and install method.
- [x] Confirm focused tests GREEN.

### Task 2: RED/GREEN Idempotency and Failure Atomicity

- [x] Add same-configuration idempotency test.
- [x] Add different-configuration rejection test.
- [x] Add Scheduler installation failure atomicity test.
- [x] Implement exact configuration storage and post-install assignment.
- [x] Confirm focused tests GREEN.

### Task 3: Runtime Boundary and Regression

- [x] Assert `LLMEngine.step()` has no install or direct publisher call.
- [x] Run Scheduler hook, source publisher, Engine publication, transaction,
  restore, and chunked-prefill suites.
- [x] Run compile, diff-check, staged-file, and runtime wiring audits.
- [x] Record explicit-install proof and automatic-runtime-disconnected boundary.
