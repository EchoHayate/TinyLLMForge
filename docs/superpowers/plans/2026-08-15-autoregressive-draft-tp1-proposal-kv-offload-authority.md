# Autoregressive Draft TP1 Proposal-KV Offload Authority Implementation Plan

> **For agentic workers:** Execute inline in the existing
> `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram` worktree. Do not create a
> worktree, dispatch subagents, stage, commit, push, stash, reset, or clean.

**Goal:** Extend the existing loaded TP1 learned-drafter gate with explicit
Proposal-KV offload configuration and fail-closed real H2D/D2H authority
evidence.

**Architecture:** Keep model loading, checkpoint/tokenizer identity, parity,
and lifecycle validation in the existing gate. Add one normalized
Proposal-KV configuration, pass it only to the learned engine, and compute
counter deltas from the nested production allocator snapshot before and
after each case.

**Tech Stack:** Python, pytest, TinyLLMForge `LLM`, Qwen3 learned drafter,
`ProposalKVResidencyManager` authority snapshots.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Keep `MAX_PROPOSAL_TOKENS=4`.
- Default-off must allocate no CPU backing and create no copy backend.
- Do not change verifier, fallback, target-KV transaction, Scheduler, n-gram,
  SAM, or native-MTP behavior.
- Do not run GPU, remote, NCCL, loaded-checkpoint, or performance workloads.
- Local tests must not claim real movement or promotion.

---

### Task 1: Configuration and Preflight Contract

**Files:**
- Modify: `tools/test_autoregressive_draft_tp1_engine_gate.py`
- Modify: `tools/autoregressive_draft_tp1_engine_gate.py`

**Interfaces:**
- Consumes: existing `_workload_configuration()`, `run_preflight()`, and
  `run_gate()`.
- Produces: normalized `proposal_kv` configuration with allocator mode,
  logical/GPU/CPU capacities, async-copy, and batch-copy fields.

- [x] Add failing tests for unchanged direct defaults, valid offload
  configuration, missing/off-range GPU capacity, and preflight-only output.
- [x] Run the focused tests and verify the new assertions fail because the
  public functions do not accept the new arguments.
- [x] Implement `_proposal_kv_configuration()` and thread its result through
  preflight, gate, engine-factory calls, and CLI arguments.
- [x] Run the focused configuration tests and verify they pass.

### Task 2: Allocator Snapshot Delta Evidence

**Files:**
- Modify: `tools/test_autoregressive_draft_tp1_engine_gate.py`
- Modify: `tools/autoregressive_draft_tp1_engine_gate.py`

**Interfaces:**
- Consumes:
  `executor.backend.proposal_kv_cache.entry_allocator`.
- Produces: per-case and merged movement/copy/replay/rematerialization
  evidence.

- [x] Extend the fake engine with deterministic before/after allocator
  snapshots and add failing delta/merge tests.
- [x] Run the tests and verify failure because the gate does not yet expose
  allocator evidence.
- [x] Add snapshot extraction, monotonic counter validation, per-case deltas,
  and capacity/mode consistency checks.
- [x] Run the focused tests and verify they pass.

### Task 3: Fail-Closed Terminal Classification

**Files:**
- Modify: `tools/test_autoregressive_draft_tp1_engine_gate.py`
- Modify: `tools/autoregressive_draft_tp1_engine_gate.py`

**Interfaces:**
- Consumes: merged allocator evidence.
- Produces:
  `proposal_kv_offload_enabled` and
  `real_proposal_kv_bidirectional_movement`.

- [x] Add failing tests for parity without movement, positive bidirectional
  movement, missing counters, allocator-mode mismatch, and nonzero accepted
  copy/replay/rematerialization.
- [x] Run the tests and verify they fail at the missing classification or
  validation boundary.
- [x] Advance the schema version, implement the classification, and update
  `validate_gate_payload()`.
- [x] Run the focused tests and verify they pass.

### Task 4: Regression and Documentation

**Files:**
- Modify:
  `docs/superpowers/audits/2026-08-15-phase1-prompt-to-artifact-coverage.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] Run the complete TP1 gate tests.
- [x] Run learned-drafter registration/backend/storage regressions.
- [x] Run changed-file `py_compile`.
- [x] Run scoped `git diff --check`.
- [x] Record the local harness classification without changing
  `REAL_AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED`,
  `PHASE_1=NOT_ACHIEVED`, or `PROMOTION=NOT_PROMOTABLE`.

## 2026-08-15 Status Reconciliation

The remaining static steps were freshly rerun against the current worktree:

```text
TP1 gate/test py_compile: PASS
TP1 gate/test/audit/handoff scoped git diff --check: PASS
```

The complete local TP1 gate contract is implemented and regression-covered,
but no GPU, NCCL, remote host, or loaded-checkpoint workload was run here.
Real bidirectional Proposal-KV movement therefore remains
`NOT_ESTABLISHED`.
