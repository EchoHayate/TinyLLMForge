# Engine Hybrid Prefix Publication Transaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide an explicit fail-closed Engine publication transaction over the five acknowledged phase APIs.

**Architecture:** Make rank-local rollback a truthful idempotent abort for rejected or unseen tickets, then add an Engine coordinator that interprets all-rank business statuses, rolls every rank back before seal, and poisons on rollback, seal, or transport uncertainty.

**Tech Stack:** Python 3.9, dataclasses, existing publication participant and Engine acknowledged phase APIs, dependency-light tests.

## Global Constraints

- Adaptive-ngram worktree only.
- No Scheduler, ModelRunner `run()`, or `LLMEngine.step()` publication wiring.
- No subagents, staging, commits, or evidence cleanup.
- Preserve exact byte-identical cache semantics.
- Do not claim production memory, quality, or speed benefit.

---

### Task 1: RED/GREEN Broadcast-Safe Rollback

**Files:**
- Modify: `tools/test_qwen35_hybrid_prefix_publication_ticket.py`
- Modify: `tinyvllm/engine/qwen35_hybrid_prefix_publication_ticket.py`

**Interfaces:**
- Produces: `rollback(payload)` returning `rolled_back` for prepared,
  rejected, already rolled-back, or unseen same-payload tickets.

- [x] Add rejected-ticket and unseen-ticket rollback tests.
- [x] Run the participant suite and observe current `error` statuses.
- [x] Implement the minimal terminal transition while preserving committed and
  different-payload rejection.
- [x] Run participant and local coordinator suites and confirm GREEN.

### Task 2: RED/GREEN Engine Publication Coordinator

**Files:**
- Create: `tinyvllm/engine/qwen35_hybrid_prefix_engine_publication.py`
- Create: `tools/test_engine_hybrid_prefix_publication_transaction.py`

**Interfaces:**
- Consumes: the five Engine phase APIs from the transport gate.
- Produces:
  `Qwen35HybridPrefixEnginePublicationCoordinator.publish(payloads) -> bool`.

- [x] Add success and prepare-rejection rollback tests.
- [x] Add prepare-error, precommit-error, and finalize-error rollback tests.
- [x] Add rollback/seal/transport poison and reuse-block tests.
- [x] Run the new suite and observe missing coordinator failure.
- [x] Implement the minimal coordinator with strict row validation and poison.
- [x] Run the new suite and confirm GREEN.

### Task 3: RED/GREEN Engine Installation

**Files:**
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/test_engine_hybrid_prefix_publication_transaction.py`

**Interfaces:**
- Produces:
  `install_qwen35_hybrid_prefix_engine_publication_coordinator()` and
  `publish_qwen35_hybrid_prefix()`.

- [x] Add type, Engine identity, idempotency, replacement, and uninstalled
  fail-closed tests.
- [x] Run the focused suite and observe missing methods.
- [x] Implement minimal Engine state, install, and explicit publish entry point.
- [x] Confirm `LLMEngine.step()` remains publication-free.

### Task 4: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-27-engine-hybrid-prefix-publication-transaction.md`

**Interfaces:**
- Produces: complete dependency-light transaction evidence and the next runtime
  boundary.

- [x] Run publication cache/participant/local coordinator/Engine transport,
  restore, acknowledgement, and owner suites.
- [x] Run focused Python compile, `git diff --check`, staged-file check, and
  `LLMEngine.step()` source audit.
- [x] Record acknowledged business-outcome atomicity and the process-crash
  limitation.
