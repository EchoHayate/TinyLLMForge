# ModelRunner Hybrid Prefix Restore Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose installed rank-local Qwen3.5 hybrid-prefix restore participants through acknowledged ModelRunner methods and validate nested per-rank results in LLMEngine.

**Architecture:** ModelRunner owns one explicitly installed participant matching its rank and returns pickle-safe operation dictionaries. LLMEngine reuses the acknowledged command channel, validates outer success plus inner ticket/rank/operation/status identity, and returns ordered inner results without enabling scheduler admission.

**Tech Stack:** Python 3.9, existing restore participant, acknowledged command channel, dependency-light AST/method tests, PyTorch CPU fixture.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not stage, commit, merge, or delete experiment evidence.
- CPU/static tests only; no CUDA, NCCL, checkpoint, or remote work.
- Do not auto-create a Qwen3.5 pool/cache/participant in this phase.
- Do not enable scheduler hybrid-prefix admission or change `LLMEngine.step()`.
- Preserve existing fire-and-forget call behavior and Qwen3/Qwen3.5 math.
- Preserve the immutable Qwen3.5 schema-v2 canonical `NO_GO`.
- Poison on malformed nested protocol data, not on valid inner miss/error.
- Do not claim performance, cache, memory, compression, or quality benefit.

---

### Task 1: RED ModelRunner Participant Methods

**Files:**
- Create: `tools/test_model_runner_hybrid_prefix_restore_methods.py`
- Modify after RED: `tinyvllm/engine/model_runner.py`

- [x] Add installation validation and one-shot tests.
- [x] Add uninstalled fail-closed tests.
- [x] Add prepare prepared/miss/error dictionary tests.
- [x] Add validate/commit/rollback success and exception tests.
- [x] Run focused tests and confirm RED.
- [x] Import participant/payload types and initialize owner field.
- [x] Implement installation and operation methods.
- [x] Run ModelRunner-focused tests and confirm GREEN.

### Task 2: RED Engine Nested Result Aggregation

**Files:**
- Modify: `tools/test_model_runner_hybrid_prefix_restore_methods.py`
- Modify after RED: `tinyvllm/engine/llm_engine.py`

- [x] Add all-rank prepared ordering test.
- [x] Add inner miss/error non-poison tests.
- [x] Add malformed/missing nested result poison tests.
- [x] Add validate/commit/rollback all-ok tests.
- [x] Add TP=1 nested validation test.
- [x] Run Engine-focused tests and confirm RED.
- [x] Implement nested-result validation and poison helper.
- [x] Implement prepare and validate/commit/rollback wrappers.
- [x] Run focused tests and confirm GREEN.

### Task 3: Regression and Completion Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run focused tests under Python 3.9 and 3.12.
- [x] Run command ack/live wiring and Qwen3.5/hybrid regressions.
- [x] Run chunked-prefill 97/1 matrix.
- [x] Run Python 3.9/3.12 `py_compile` and `git diff --check`.
- [x] Confirm staged files empty and experiment evidence present.
- [x] Build prompt-to-artifact checklist for installation, rank identity,
  nested result validation, miss/error semantics, poison, CPU scope, and no
  performance overclaim.
- [x] Update handoff with Engine coordinator composition as next gate.
- [x] Mark only freshly verified checkboxes complete.
