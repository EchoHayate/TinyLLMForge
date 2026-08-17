# Engine Live Hybrid Prefix Restore Transaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an explicit Engine-side Qwen3.5 hybrid-prefix restore transaction that privately reserves complete KV and central hybrid-state ownership, orchestrates acknowledged all-rank prepare/validate/commit, and fails stop on uncertain cleanup or post-publication failure.

**Architecture:** A focused live coordinator owns transaction state and calls existing LLMEngine acknowledged restore wrappers while Scheduler remains fail-closed. Complete KV reservation and the allocator lease stay private until all-rank validation; publication attaches Sequence metadata, after which any commit uncertainty poisons the runtime instead of attempting unsafe reuse.

**Tech Stack:** Python 3.9, existing BlockManager reservation API, HybridStateSlotAllocator, Qwen3.5 restore payload, LLMEngine acknowledged command channel, dependency-light CPU tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not stage, commit, merge, or delete experiment evidence.
- CPU/static tests only; no CUDA, NCCL, checkpoint, local GPU, or remote GPU work.
- Do not enable Scheduler hybrid-prefix admission or change `LLMEngine.step()`.
- Do not automatically construct Qwen3.5 pools, caches, or participants.
- Preserve ordinary `model_runner.call()` behavior and existing model math.
- Preserve the immutable Qwen3.5 schema-v2 canonical `NO_GO`.
- Keep complete KV and allocator ownership private until all-rank validation.
- Poison after uncertain cleanup or any post-publication failure.
- Do not claim performance, cache, memory, compression, or quality benefit.

---

### Task 1: RED Live Coordinator Reservation and Prepare

**Files:**
- Create: `tools/test_engine_live_hybrid_prefix_restore_transaction.py`
- Create after RED: `tinyvllm/engine/qwen35_hybrid_prefix_engine_restore.py`

**Interfaces:**
- Consumes: `BlockManager.reserve_sequence_blocks()`, `HybridStateSlotAllocator.allocate()`, `Qwen35HybridPrefixRestorePayload`, and `LLMEngine.prepare_model_runner_hybrid_prefix_restore()`.
- Produces: `Qwen35HybridPrefixEngineRestoreTicket` and `Qwen35HybridPrefixEngineRestoreCoordinator.acquire()`.

- [x] Write constructor, request-validation, exact-prefix-miss, and private-reservation tests.
- [x] Write all-rank prepared and valid miss/error cleanup tests.
- [x] Run the focused test and confirm RED because the live coordinator module is absent.
- [x] Implement ticket states, constructor validation, private complete reservation, lease allocation, payload construction, and prepare orchestration.
- [x] Make participant rollback idempotent only for the exact already-rolled-back ticket.
- [x] Run the focused test and restore-ticket regression; confirm GREEN.

### Task 2: RED Precommit, Publication, Commit, and Poison

**Files:**
- Modify: `tools/test_engine_live_hybrid_prefix_restore_transaction.py`
- Modify after RED: `tinyvllm/engine/qwen35_hybrid_prefix_engine_restore.py`

**Interfaces:**
- Consumes: the Task 1 ticket plus Engine validate/rollback/commit wrappers.
- Produces: validated publication ordering, terminal success/failure states, and fail-stop poison behavior.

- [x] Add stale allocator, dirty Sequence, stale reservation, and validate-failure tests.
- [x] Add rollback failure poison and future-acquire rejection tests.
- [x] Add publication-order and post-publication commit-failure tests.
- [x] Run focused tests and confirm the new cases RED for missing transaction phases.
- [x] Implement local precommit validation and all-rank validate.
- [x] Implement deterministic pre-publication cleanup and poison on uncertain cleanup.
- [x] Implement KV attach plus Sequence lease publication.
- [x] Implement all-rank commit and post-publication fail-stop poison.
- [x] Run focused tests and confirm GREEN.

### Task 3: RED Explicit LLMEngine Installation and Delegation

**Files:**
- Modify: `tools/test_engine_live_hybrid_prefix_restore_transaction.py`
- Modify after RED: `tinyvllm/engine/llm_engine.py`

**Interfaces:**
- Consumes: `Qwen35HybridPrefixEngineRestoreCoordinator`.
- Produces: `install_qwen35_hybrid_prefix_engine_restore_coordinator()` and `acquire_qwen35_hybrid_prefix()`.

- [x] Add exact type, Engine/BlockManager/allocator identity, one-shot, and uninstalled fail-closed tests.
- [x] Add installed acquisition delegation test.
- [x] Add static assertions that `LLMEngine.step()` and Scheduler guard do not invoke the coordinator.
- [x] Run focused tests and confirm RED for missing Engine methods.
- [x] Import and initialize the optional coordinator.
- [x] Implement explicit installation validation and acquisition delegation.
- [x] Run focused tests and confirm GREEN.

### Task 4: Regression and Completion Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

**Interfaces:**
- Consumes: all prior task artifacts.
- Produces: fresh validation evidence and the next blocked runtime gate.

- [x] Run the focused live transaction tests under Python 3.9 and 3.12.
- [x] Run restore-ticket, ModelRunner restore methods, command ack, and live wiring tests.
- [x] Run the Qwen3.5/hybrid CPU regression set.
- [x] Run the chunked-prefill function matrix and verify 97 pass / 1 known skip / 0 fail.
- [x] Run Python 3.9/3.12 `py_compile` and `git diff --check`.
- [x] Confirm staged files are empty and untracked `experiments/` evidence remains.
- [x] Audit every spec acceptance item and claim boundary.
- [x] Update handoff with automatic owner construction/Scheduler/GPU gates still blocked.
- [x] Mark only freshly verified plan checkboxes complete.

