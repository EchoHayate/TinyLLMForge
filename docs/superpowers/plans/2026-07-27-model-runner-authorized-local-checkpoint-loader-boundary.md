# ModelRunner Authorized Local Checkpoint Loader Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-off ModelRunner boundary that accepts a bounded authorized checkpoint-load request, invokes an explicitly installed local loader, and atomically publishes only a fully validated candidate.

**Architecture:** Define a frozen exact request type, install one local loader with a matching authorization digest, and retain aggregate completion only after the existing one-shot candidate slot publishes successfully. Expected failures return bounded error rows and leave state pristine.

**Tech Stack:** Python dataclasses, existing checkpoint candidate and publication slot, dependency-light AST tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not implement or execute the real remote checkpoint worker.
- Do not open a checkpoint payload, initialize CUDA, or run SSH.
- Do not import `tools` gate modules into production runtime.
- Keep Engine and `LLMEngine.step()` disconnected.
- Preserve schema-v2 canonical `NO_GO`.
- Do not stage, commit, merge, or clean experiment evidence.
- Do not claim production memory, compression, quality, or speed benefit.

---

### Task 1: Bounded Load Request

**Files:**
- Create: `tinyvllm/models/qwen35_checkpoint_worker.py`
- Create: `tools/test_qwen35_checkpoint_worker_request.py`

- [x] Write RED tests for exact type, absolute bounded path, SHA256 fields, and positive budget.
- [x] Run focused tests and confirm the module is absent.
- [x] Implement the frozen request contract.
- [x] Run focused tests and confirm GREEN.

### Task 2: ModelRunner Loader Installation and Atomic Publication

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Create: `tools/test_model_runner_authorized_checkpoint_loader.py`

- [x] Write RED tests for default-off state, installation idempotency/conflict, authorization mismatch, loader failure, invalid candidate, success, exact repeat, and conflicting repeat.
- [x] Run focused tests and confirm missing methods fail.
- [x] Implement minimal explicit installation and local load/publish methods.
- [x] Run focused tests and confirm GREEN.

### Task 3: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan

- [x] Run request, local loader, candidate publication, all-rank binding, checkpoint loading, owner, identity, and publication-runtime regressions.
- [x] Run compile, `git diff --check`, staged-file, payload-access source, Engine, and `step()` audits.
- [x] Check every plan item and record exact evidence plus the remaining production factory/worker gap.
