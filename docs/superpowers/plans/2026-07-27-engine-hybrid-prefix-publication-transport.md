# Engine Hybrid Prefix Publication Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Broadcast rank-bound publication payload matrices through the acknowledged command channel and expose five strict Engine phase APIs.

**Architecture:** Engine validates one payload per TP rank and broadcasts the complete immutable matrix. Each ModelRunner selects its own rank payload before delegating to the existing participant method, and Engine validates one exact nested result per rank.

**Tech Stack:** Python 3.9, dataclasses, existing shared-memory command acknowledgement channel, dependency-light AST tests.

## Global Constraints

- Adaptive-ngram worktree only.
- No automatic publication transaction, Scheduler wiring, or `LLMEngine.step()` call.
- No subagents, staging, commits, or evidence cleanup.
- Preserve exact publication participant semantics and valid business statuses.
- Do not claim production memory or speed benefit.

---

### Task 1: RED Ranked ModelRunner Payload Selection

**Files:**
- Modify: `tools/test_model_runner_hybrid_prefix_publication_methods.py`
- Modify: `tinyvllm/engine/model_runner.py`

**Interfaces:**
- Consumes: existing five rank-local publication methods.
- Produces: `_qwen35_hybrid_prefix_publication_payload(payload_or_payloads)`.

- [x] Add tests that pass a two-rank payload tuple and require rank-local
  delegation of only the matching row.
- [x] Add malformed matrix tests for missing, duplicate, and wrong-rank rows.
- [x] Run the focused test and observe failure because matrix selection is
  absent.
- [x] Implement the minimal selection helper and call it from all five methods.
- [x] Run the focused test and confirm GREEN.

### Task 2: RED/GREEN Engine Payload and Result Validation

**Files:**
- Create: `tools/test_engine_hybrid_prefix_publication_transport.py`
- Modify: `tinyvllm/engine/llm_engine.py`

**Interfaces:**
- Consumes: complete payload matrix and existing
  `call_model_runner_acknowledged`.
- Produces:
  `_validate_hybrid_prefix_publication_payloads(payloads)` and
  `_validate_hybrid_prefix_publication_results(...)`.

- [x] Add tests for exact matrix identity, TP size, participant coverage, and
  pre-dispatch rejection.
- [x] Add tests for ordered outer-rank/inner-participant aggregation and exact
  nested result fields.
- [x] Add malformed nested result tests that require collector poisoning.
- [x] Run the new test and observe missing helper failures.
- [x] Implement minimal payload/result validators.
- [x] Run the new test and confirm validator GREEN.

### Task 3: RED/GREEN Five Engine Phase APIs

**Files:**
- Modify: `tools/test_engine_hybrid_prefix_publication_transport.py`
- Modify: `tinyvllm/engine/llm_engine.py`

**Interfaces:**
- Produces:
  `prepare_model_runner_hybrid_prefix_publication`,
  `precommit_model_runner_hybrid_prefix_publication`,
  `finalize_model_runner_hybrid_prefix_publication`,
  `seal_model_runner_hybrid_prefix_publication`, and
  `rollback_model_runner_hybrid_prefix_publication`.

- [x] Add exact method-name, timeout, allowed-status, and ordered-row tests for
  all five phases.
- [x] Run the new test and observe missing phase API failures.
- [x] Implement one generic phase helper plus the five explicit public methods.
- [x] Confirm valid `rejected` and `error` rows are preserved without poisoning.
- [x] Confirm `LLMEngine.step()` remains publication-free.

### Task 4: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-27-engine-hybrid-prefix-publication-transport.md`

**Interfaces:**
- Produces: exact validation evidence and the next transaction-coordinator
  boundary.

- [x] Run publication cache/participant/coordinator, ModelRunner methods,
  command acknowledgement, live restore, and owner suites.
- [x] Run focused Python compile, `git diff --check`, and staged-file check.
- [x] Record that phase transport is proven but distributed transaction
  orchestration and automatic runtime publication remain disconnected.
