# Qwen3.5 Runtime Prefix Publication Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture immutable exact full-prompt publication candidates after completed aligned prefill, without publishing them.

**Architecture:** A focused capture module validates Sequence, BlockManager, and HybridStateSlotAllocator identity at one point in time, freezes exact token/block/lease metadata, and constructs a rank payload matrix on demand.

**Tech Stack:** Python 3.9, frozen dataclasses, existing Sequence/BlockManager/hybrid lease/publication payload types, dependency-light CPU tests.

## Global Constraints

- Adaptive-ngram worktree only.
- No Scheduler postprocess, `LLMEngine.step()`, or ModelRunner `run()` wiring.
- No subagents, staging, commits, or evidence cleanup.
- Reject non-aligned prompts; never pair truncated tokens with full-prompt state.
- Do not claim production memory or speed benefit.

---

### Task 1: RED/GREEN Exact Candidate Capture

**Files:**
- Create: `tinyvllm/engine/qwen35_hybrid_prefix_publication_candidate.py`
- Create: `tools/test_qwen35_hybrid_prefix_publication_candidate.py`

**Interfaces:**
- Produces:
  `capture_qwen35_hybrid_prefix_publication_candidate(...)` and
  `Qwen35HybridPrefixPublicationCandidate`.

- [x] Add an aligned completed-prefill fixture with a live allocator lease.
- [x] Add exact frozen token, block identity, key, and lease assertions.
- [x] Run the new suite and observe missing module failure.
- [x] Implement minimal validation and immutable capture.
- [x] Run the new suite and confirm GREEN.

### Task 2: RED/GREEN Fail-Closed Eligibility

**Files:**
- Modify: `tools/test_qwen35_hybrid_prefix_publication_candidate.py`
- Modify: `tinyvllm/engine/qwen35_hybrid_prefix_publication_candidate.py`

**Interfaces:**
- Preserves: exact full-prompt state/token alignment.

- [x] Add non-aligned and incomplete prefill rejection tests.
- [x] Add missing hash, stale generation, stale block tokens, and released lease
  rejection tests.
- [x] Add post-capture Sequence/Block mutation-isolation tests.
- [x] Implement only the validations required by those tests.
- [x] Confirm the focused suite remains GREEN.

### Task 3: RED/GREEN Payload Matrix

**Files:**
- Modify: `tools/test_qwen35_hybrid_prefix_publication_candidate.py`
- Modify: `tinyvllm/engine/qwen35_hybrid_prefix_publication_candidate.py`

**Interfaces:**
- Produces:
  `candidate.publication_payloads(ticket_id, world_size)`.

- [x] Add exact contiguous participant matrix and pickle round-trip tests.
- [x] Add invalid ticket/world-size tests.
- [x] Implement minimal payload construction.
- [x] Confirm the focused suite and participant/transport suites pass.

### Task 4: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-runtime-prefix-publication-candidate.md`

**Interfaces:**
- Produces: eligibility evidence and the next lifetime-pinning boundary.

- [x] Run candidate, publication stack, Engine transaction, restore, and hybrid
  state suites.
- [x] Run focused Python compile, `git diff --check`, staged-file check, and
  runtime source audit.
- [x] Record that candidates do not pin source KV/state lifetime and automatic
  publication remains disconnected.
