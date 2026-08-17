# Qwen3.5 Real Checkpoint Load Run Authorization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local-only source-clean, source-bound authorization decision between preflight and future worker implementation.

**Architecture:** Validate local preflight/source artifacts, compare all recorded hash maps to current owned-source hashes, independently inspect Git tracked/staged/unstaged/untracked state, and emit `BLOCKED` or `AUTHORIZED`. Wire this as `authorization-only` without SSH, worker launch, or payload access.

**Tech Stack:** Python 3 standard library, Git read-only commands, JSON, SHA256, executable dependency-light tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not implement or launch the worker.
- Do not SSH, open checkpoint payloads, kill GPU processes, switch GPUs, or weaken `gpu0_idle`.
- Do not stage, commit, merge, or delete experiment evidence.
- `worker_execution_authorized` remains false in every result.

---

### Task 1: Authorization Contract RED Tests

**Files:**
- Create: `tools/qwen35_real_checkpoint_load_authorization.py`
- Create: `tools/test_qwen35_real_checkpoint_load_authorization.py`

- [x] Write synthetic `AUTHORIZED` and `BLOCKED` tests.
- [x] Cover status, source hashes, branch/commit, and Git-clean failures.
- [x] Run tests and confirm failure because the authorization API is absent.

### Task 2: Minimal Local Authorization Verifier

**Files:**
- Modify: `tools/qwen35_real_checkpoint_load_authorization.py`
- Test: `tools/test_qwen35_real_checkpoint_load_authorization.py`

- [x] Implement JSON loading, preflight validation, source-map validation, and
      current source hashing.
- [x] Implement read-only Git tracked/staged/unstaged/untracked inspection for
      the frozen owned-source paths.
- [x] Emit explicit checks/reasons and never throw for ordinary blocked
      evidence.
- [x] Run tests to GREEN.

### Task 3: Runner `authorization-only` Mode

**Files:**
- Modify: `tools/run_qwen35_real_checkpoint_load_gate_remote.py`
- Modify: `tools/test_qwen35_real_checkpoint_load_safety_gate.py`

- [x] Add RED tests for mode allowance, orchestration, output persistence, and
      no SSH/worker/payload behavior.
- [x] Dynamically load and invoke the local authorization module.
- [x] Keep `run` as the sole intentional rejection path.
- [x] Run safety and authorization tests to GREEN.

### Task 4: Live Local Decision And Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-real-checkpoint-load-run-authorization.md`

- [x] Run `authorization-only` against the current-source preflight.
- [x] Require `BLOCKED` for both GPU occupancy and dirty/untracked owned source.
- [x] Run focused tests, `py_compile`, `git diff --check`, staged-file check,
      source-binding check, and EOF handoff check.
- [x] Record that the long-term goal remains active.
