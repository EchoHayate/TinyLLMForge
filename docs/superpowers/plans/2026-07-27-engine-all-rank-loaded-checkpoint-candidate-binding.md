# Engine All-Rank Loaded Checkpoint Candidate Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explicitly bind one locally published verified Qwen3.5 checkpoint candidate on every ModelRunner rank through a zero-payload acknowledged Engine command.

**Architecture:** Extend the existing one-shot publication slot to retain the exact candidate, add ModelRunner local publish and non-throwing bind-result methods, then aggregate and validate all rank rows in `LLMEngine`. Participant rejection remains exact-retryable; uncertain command transport remains fail-stop.

**Tech Stack:** Python, existing candidate publication slot, ModelRunner acknowledged command transport, dependency-light AST tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Never serialize or broadcast a loaded model candidate through shared memory.
- Preserve candidate owner and verified manifest SHA256 provenance exactly.
- Keep `LLMEngine.__init__()` and `LLMEngine.step()` free of automatic binding.
- Preserve acknowledgement transport fail-stop semantics.
- Do not stage, commit, merge, or clean untracked experiment evidence.
- Do not claim production memory, compression, quality, or speed benefit.

---

### Task 1: Local One-Shot Candidate Handoff

**Files:**
- Modify: `tinyvllm/engine/qwen35_hybrid_model_publication.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_qwen35_hybrid_model_publication.py`
- Create: `tools/test_model_runner_published_checkpoint_candidate_binding.py`

**Interfaces:**
- Produces: `Qwen35HybridModelOwnerPublicationSlot.candidate`.
- Produces: `ModelRunner.publish_qwen35_loaded_checkpoint_candidate(candidate)`.
- Produces: `ModelRunner.bind_published_qwen35_loaded_checkpoint_candidate()`.

- [x] Write RED tests for atomic candidate retention, local publication, missing-candidate error row, first bind, and exact repeat.
- [x] Run focused tests and confirm missing properties/methods fail.
- [x] Implement the minimal slot and ModelRunner methods.
- [x] Run focused tests and confirm GREEN.

### Task 2: Engine All-Rank Aggregation and Retry

**Files:**
- Modify: `tinyvllm/engine/llm_engine.py`
- Create: `tools/test_engine_all_rank_loaded_checkpoint_candidate_binding.py`

**Interfaces:**
- Consumes: zero-argument ModelRunner bind-result method from Task 1.
- Produces: `LLMEngine.bind_qwen35_loaded_checkpoint_candidates(*, timeout_s)`.

- [x] Write RED tests for zero-payload dispatch, rank schema, homogeneous provenance, participant error, exact retry, completed idempotency, and conflict closure.
- [x] Run focused tests and confirm the missing Engine method fails.
- [x] Implement minimal acknowledged aggregation and completed-state storage.
- [x] Run focused tests and confirm GREEN.

### Task 3: Default-Off Audit and Regression

**Files:**
- Modify: `tools/test_engine_all_rank_loaded_checkpoint_candidate_binding.py`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan

**Interfaces:**
- Consumes: completed explicit all-rank binding.
- Produces: verified claim boundary and next worker-integration TODO.

- [x] Assert `LLMEngine.step()` contains no all-rank candidate-binding reference.
- [x] Run candidate, publication-slot, owner, identity, command-ack, Engine publication-runtime, restore, and publication regressions.
- [x] Run focused compile, `git diff --check`, staged-file, and payload-size/source audits.
- [x] Check every plan item and append exact evidence plus remaining real-worker gap to `AGENT_HANDOFF_STATE.md`.
