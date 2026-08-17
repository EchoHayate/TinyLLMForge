# Qwen3.5 Live Restore Owner Factory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically derive rank-local Qwen3.5 hybrid-prefix restore owners from each ModelRunner's already installed runtime-bridge pool, validate all-rank owner identity, and install the Engine transaction coordinator without allocating duplicate state tensors.

**Architecture:** A focused pure factory builds adapters, transaction, cache, and participant around the exact existing pool. ModelRunner exposes an acknowledged idempotent configuration method; LLMEngine validates ordered all-rank identity rows before constructing the already tested central coordinator.

**Tech Stack:** Python 3.9, PyTorch CPU tensors, existing HybridStateTensorPool/runtime bridge, acknowledged ModelRunner command channel, dependency-light AST/method tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not stage, commit, merge, or delete experiment evidence.
- CPU/static tests only; no CUDA, NCCL, checkpoint, local GPU, or remote GPU work.
- Do not construct a second state pool or runtime bridge.
- Do not claim current `Qwen3ForCausalLM` is a native Qwen3.5 owner.
- Do not enable Scheduler hybrid-prefix admission or change `LLMEngine.step()`.
- Preserve ordinary `model_runner.call()` and existing model math.
- Preserve the immutable Qwen3.5 schema-v2 canonical `NO_GO`.
- Poison malformed or inconsistent all-rank owner identity.
- Do not claim performance, cache, memory, compression, or quality benefit.

---

### Task 1: RED Pure Rank-Local Owner Factory

**Files:**
- Create: `tools/test_qwen35_live_restore_owner_factory.py`
- Create after RED: `tinyvllm/engine/qwen35_hybrid_prefix_owner.py`

**Interfaces:**
- Consumes: one existing `HybridStateTensorPool`, participant rank, and cache limits.
- Produces: `Qwen35HybridPrefixRestoreOwner` and `build_qwen35_hybrid_prefix_restore_owner()`.

- [x] Write exact object-graph, adapter ordering, and no-new-storage tests.
- [x] Write malformed/incomplete/empty layout and invalid limit/rank tests.
- [x] Run focused tests and confirm RED because the owner module is absent.
- [x] Implement strict layer-role pairing and owner construction.
- [x] Run pure factory tests and confirm GREEN.

### Task 2: RED ModelRunner Owner Configuration

**Files:**
- Modify: `tools/test_qwen35_live_restore_owner_factory.py`
- Modify after RED: `tinyvllm/engine/model_runner.py`

**Interfaces:**
- Consumes: installed `hybrid_state_runtime_bridge`.
- Produces: `configure_qwen35_hybrid_prefix_restore_owner(max_entries, max_bytes) -> dict`.

- [x] Add missing-bridge fail-closed and exact participant-install tests.
- [x] Add exact pickle-safe identity-row tests.
- [x] Add identical idempotence and changed-limit/replacement rejection tests.
- [x] Run focused tests and confirm RED for missing ModelRunner method.
- [x] Import owner factory, initialize retained owner field, and implement configuration.
- [x] Run focused tests and confirm GREEN.

### Task 3: RED Engine All-Rank Configuration

**Files:**
- Modify: `tools/test_qwen35_live_restore_owner_factory.py`
- Modify after RED: `tinyvllm/engine/llm_engine.py`

**Interfaces:**
- Consumes: acknowledged ModelRunner owner configuration and Scheduler-owned allocator/BlockManager.
- Produces: `configure_qwen35_hybrid_prefix_restore(...)`.

- [x] Add TP=1 and ordered TP>1 success tests.
- [x] Add inner-vs-outer rank, missing rank, malformed fields, and cross-rank mismatch poison tests.
- [x] Add allocator-capacity mismatch and missing-allocator tests.
- [x] Add identical Engine idempotence and changed-settings rejection tests.
- [x] Add static guard/step non-integration assertions.
- [x] Run focused tests and confirm RED for missing Engine factory.
- [x] Implement all-rank identity validation and poison helper reuse.
- [x] Construct/install the central coordinator only after successful validation.
- [x] Run focused tests and confirm GREEN.

### Task 4: Regression and Completion Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

**Interfaces:**
- Consumes: all factory/configuration artifacts.
- Produces: fresh evidence and the next native-model/runtime-bridge gate.

- [x] Run focused owner-factory tests under Python 3.9 and 3.12.
- [x] Run live transaction, restore-ticket, ModelRunner methods, ack, and wiring tests.
- [x] Run Qwen3.5/hybrid CPU regressions.
- [x] Run chunked-prefill 97/1/0 function matrix.
- [x] Run Python 3.9/3.12 `py_compile` and `git diff --check`.
- [x] Confirm staged files empty and experiment evidence present.
- [x] Audit every spec requirement and claim boundary.
- [x] Update handoff with native Qwen3.5 model/runtime bridge and Scheduler gates still blocked.
- [x] Mark only freshly verified checkboxes complete.

