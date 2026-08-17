# Qwen3.5 Hybrid Prefix Restore Ticket Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CPU-tested two-phase Engine↔ModelRunner restore-ticket protocol that publishes hybrid-prefix KV and state metadata only after every modeled participant prepares successfully.

**Architecture:** Keep complete KV reservations and state-slot allocation in an Engine-side coordinator while rank-local participants own tensor pools and snapshot restore. Exchange an immutable pickle-safe payload and explicit prepare acknowledgements; roll back every participant and Engine-owned resource on any miss or error.

**Tech Stack:** Python 3.9, dataclasses, pickle, PyTorch CPU tensors, existing BlockManager, hybrid-state allocator/pool, Qwen3.5 snapshot cache, dependency-light script tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not stage, commit, merge, or delete untracked experiment evidence.
- CPU-only; do not start local or remote GPU/checkpoint work.
- Do not modify `LLMEngine.step()`, live worker RPC, or scheduler admission.
- Do not move tensor pools or snapshots into Scheduler.
- Do not modify Qwen3 math, attention, RoPE, RMSNorm, or checkpoint semantics.
- Preserve the immutable Qwen3.5 schema-v2 canonical `NO_GO`.
- Exact token comparison is mandatory in addition to chained hashes.
- Publish no request KV or hybrid-state metadata before all participants prepare.
- Roll back every prepared participant, allocator lease, and KV reservation on miss or error.
- Do not claim latency, throughput, hit rate, compression, quality, or physical-memory improvement.

---

### Task 1: RED Restore Payload and Participant Protocol

**Files:**
- Create: `tools/test_qwen35_hybrid_prefix_restore_ticket.py`
- Create after RED: `tinyvllm/engine/qwen35_hybrid_prefix_restore_ticket.py`

**Interfaces:**
- Produces: `Qwen35HybridPrefixRestorePayload`.
- Produces: `Qwen35HybridPrefixPrepareAck`.
- Produces: `Qwen35HybridPrefixRestoreParticipant`.

- [x] Add a focused fixture with real rank-local pools, adapters,
  transactions, and snapshot caches.
- [x] Add payload and acknowledgement pickle round-trip tests.
- [x] Add participant prepare tests for exact restore and idempotent duplicate
  prepare.
- [x] Add participant miss and restore-error tests proving no retained pool
  binding.
- [x] Add conflicting payload, stale lease, rollback, commit, and repeated
  terminal-operation tests.
- [x] Run the focused script and confirm RED because the restore-ticket module
  does not exist.
- [x] Implement immutable payload and acknowledgement validation.
- [x] Implement participant prepare/validate/commit/rollback with explicit
  `prepared`, `miss`, and `error` acknowledgement.
- [x] Run participant-focused tests and confirm GREEN.

### Task 2: RED Engine-Owned Ticket Coordinator

**Files:**
- Modify: `tools/test_qwen35_hybrid_prefix_restore_ticket.py`
- Modify after RED: `tinyvllm/engine/qwen35_hybrid_prefix_restore_ticket.py`

**Interfaces:**
- Produces: `Qwen35HybridPrefixRestoreTicket`.
- Produces: `Qwen35HybridPrefixRestoreCoordinator.reserve()`.
- Produces: `Qwen35HybridPrefixRestoreCoordinator.prepare()`.
- Produces: `Qwen35HybridPrefixRestoreCoordinator.commit()`.
- Produces: `Qwen35HybridPrefixRestoreCoordinator.rollback()`.

- [x] Add reserve tests proving complete KV plus allocator ownership remains
  private and the request is pristine.
- [x] Add clean shorter-prefix miss and reserve-exception rollback tests.
- [x] Add exact-token, key, TP-size, participant-ID, and dirty-request
  validation tests.
- [x] Add all-prepared multi-participant transition tests.
- [x] Add participant miss and error tests proving rollback of earlier ranks,
  allocator lease, complete KV reservation, and request metadata.
- [x] Add explicit rollback tests from `reserved` and `prepared`.
- [x] Add commit tests proving KV and lease metadata appear only after all
  participants prepare.
- [x] Add stale allocator, participant, reservation, and destination
  precommit tests proving no publication and explicit rollback remains valid.
- [x] Add invalid transition and repeated terminal-operation tests.
- [x] Run coordinator-focused tests and confirm RED because coordinator
  interfaces do not exist.
- [x] Implement strict ticket and coordinator validation.
- [x] Implement reserve with exact prefix coverage and all-resource rollback.
- [x] Implement prepare with stable participant order and acknowledgement
  collection.
- [x] Implement prevalidated commit and explicit rollback.
- [x] Run the focused script and confirm GREEN.

### Task 3: Regression and Completion Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run the focused restore-ticket script.
- [x] Run the full Python 3.12 zero-argument
  `tools/test_chunked_prefill.py` matrix and confirm only the documented Config
  AST skip.
- [x] Run hybrid prefix acquisition/cache, cross-layer transaction, layer
  adapter, hybrid-state allocator/sequence/scheduler/runtime-bridge, packed
  layer-stack, and ModelRunner dependency-light regressions.
- [x] Run Python 3.9 and Python 3.12 `py_compile` for the new module and test.
- [x] Run `git diff --check`.
- [x] Confirm `git diff --cached --name-only` is empty.
- [x] Confirm no tracked or untracked `experiments/` evidence was removed.
- [x] Build a prompt-to-artifact checklist covering exact identity, private
  reservation, explicit participant acknowledgement, all-participant gate,
  rollback, atomic publication boundary, CPU-only scope, and no performance
  overclaim.
- [x] Update `AGENT_HANDOFF_STATE.md` with fresh commands, results, allowed
  conclusion, and the real worker acknowledgement/barrier as the next gate.
- [x] Mark checkboxes complete only from fresh evidence.
