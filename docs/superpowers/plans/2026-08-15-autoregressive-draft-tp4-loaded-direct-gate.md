# Autoregressive Draft TP4 Loaded Direct Gate Implementation Plan

> **For agentic workers:** Execute inline in the existing
> `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram` worktree. Do not create a
> worktree, dispatch subagents, stage, commit, push, stash, reset, or clean.

**Goal:** Build a runnable direct-allocator TP4 loaded gate for Qwen3
independent drafting against a Qwen3.5 target.

**Architecture:** Reuse TP1 prompt/workload/identity helpers and the existing
TP4 rank-snapshot transport/validator. Keep the artifact independent from
native-MTP-specific campaign receipts and explicitly retain no-movement,
no-performance, and no-promotion boundaries.

**Tech Stack:** Python, pytest, TinyLLMForge `LLM`,
`EngineSpeculativeRuntime`, TP4 ModelRunner acknowledgements.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Keep `MAX_PROPOSAL_TOKENS=4`.
- The gate is direct allocator only.
- Do not change runtime, Scheduler, verifier, target-KV, n-gram, SAM, or MTP
  behavior.
- Do not run GPU, remote, NCCL, loaded-checkpoint, or performance workloads.
- Do not claim real movement, performance, promotion, or Phase 1 completion.

---

### Task 1: Gate Payload and Validation

**Files:**
- Create: `tools/autoregressive_draft_tp4_engine_gate.py`
- Create: `tools/test_autoregressive_draft_tp4_engine_gate.py`

- [x] Write failing fake-engine tests for lifecycle order, TP4/direct
  configuration, batch-1/4 exact parity, acceptance rows, and rank summaries.
- [x] Run the tests and verify failure because the gate module is absent.
- [x] Implement dependency-injected `run_gate()` and fail-closed
  `validate_gate_payload()`.
- [x] Run the focused tests and verify they pass.

### Task 2: Production Adapter and CLI

**Files:**
- Modify: `tools/autoregressive_draft_tp4_engine_gate.py`
- Modify: `tools/test_autoregressive_draft_tp4_engine_gate.py`

- [x] Add failing tests for GPU/port validation, environment restoration,
  production engine kwargs, registration activation, and atomic output.
- [x] Run the tests and verify the missing adapter/CLI behavior fails.
- [x] Implement `_TinyVLLMTP4EngineAdapter`, distributed environment handling,
  prompt-file CLI parsing, and exclusive JSON output.
- [x] Run the complete gate tests.

### Task 3: Regression and Documentation

**Files:**
- Modify:
  `docs/superpowers/audits/2026-08-15-phase1-prompt-to-artifact-coverage.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] Run TP4 gate, transport, validator, and learned-drafter regressions.
- [x] Run changed-file `py_compile`.
- [x] Run scoped `git diff --check`.
- [x] Record the local gate classification while retaining loaded parity,
  movement, performance, Phase 1, and promotion as not established.
