# Autoregressive Draft Proposal-Forward Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add source-bound proposal-forward substage evidence and run a stable repeated TP4 learned-b4 diagnostic before choosing the first performance optimization.

**Architecture:** Preserve the existing executor parent timing and add a nested six-key, non-synchronizing detail mapping. Extend the worker, schema, aggregate, verifier, and remote runner to carry exact rank deltas, then run a focused two-warmup/eight-repeat learned-b4 campaign paired with a fresh target reference.

**Tech Stack:** Python 3.11, pytest, PyTorch CUDA, TinyLLMForge Engine API, Bash, SSH.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Keep `max_proposal_tokens=4`, temperature zero, exact parity, and workload-derived Proposal-KV capacity.
- Do not add CUDA synchronization to the measured request path.
- Use `sitian@10.232.195.203` and GPUs `3,4,6,7`.
- Do not terminate unrelated GPU processes.
- Do not stage, commit, push, switch branches, stash, reset, or clean.

---

### Task 1: Freeze Executor Detail Timing

**Files:**
- Modify: `tools/test_autoregressive_draft_executor.py`
- Modify: `tinyvllm/engine/autoregressive_draft_executor.py`

**Interfaces:**
- Produces: `authority_snapshot()["proposal_forward_detail_ms"]`.
- Keys: `setup`, `backend_submit`, `selection_collective`,
  `decode_authority`, `token_readback`, `materialize_register`.

- [ ] Add a deterministic-clock test that runs one exact-Q proposal and
  expects all six keys with exact accumulated values.
- [ ] Run the focused test and confirm RED because the detail mapping is
  absent.
- [ ] Add the six counters and non-overlapping timers without changing model,
  allocator, TP, or lifecycle semantics.
- [ ] Re-run the focused test and the full executor file remotely.

### Task 2: Extend Worker and Schema-v3 Contract

**Files:**
- Modify: `tools/test_autoregressive_draft_performance_gate.py`
- Modify: `tools/autoregressive_draft_performance_worker.py`
- Modify: `tools/autoregressive_draft_performance_gate.py`
- Modify: `tools/verify_autoregressive_draft_performance_gate.py`

**Interfaces:**
- Produces per run:
  `runtime.draft_executor_proposal_detail`.
- Produces aggregate metrics:
  `executor_detail_<key>_ms`,
  `executor_detail_sum_ms`,
  `executor_detail_residual_ms`.

- [ ] Add RED tests for four-rank before/after delta extraction, target-zero
  semantics, learned-positive semantics, aggregate recomputation, negative
  residual rejection, and detail tamper rejection.
- [ ] Raise `SCHEMA_VERSION` from 2 to 3.
- [ ] Implement exact six-key extraction and max-rank aggregation.
- [ ] Re-run the performance gate tests and `py_compile`.

### Task 3: Add Focused Batch-4 Diagnostic

**Files:**
- Modify: `tools/autoregressive_draft_performance_worker.py`
- Create: `tools/autoregressive_draft_b4_timing_diagnostic.py`
- Create: `tools/verify_autoregressive_draft_b4_timing_diagnostic.py`
- Create: `tools/run_autoregressive_draft_b4_timing_diagnostic_remote.sh`
- Modify: `tools/test_autoregressive_draft_performance_gate.py`

**Interfaces:**
- Worker accepts explicit `--warmup-runs` and `--measured-runs`.
- Diagnostic consumes one target-b4 worker and one learned-b4 worker.
- Diagnostic emits raw repeats, parity, timing stationarity rows, source
  hashes, classification, and verifier receipt.

- [ ] Add RED tests for two warmups, eight measured runs, exact parity,
  raw-detail retention, stationarity classification, source drift rejection,
  bounded timeout, and dual verifier calls.
- [ ] Implement worker run-count arguments while preserving default `1/3`.
- [ ] Implement the focused assembler, independent verifier, and remote runner.
- [ ] Run focused tests, `py_compile`, and `bash -n`.

### Task 4: Execute and Choose the Optimization

**Files:**
- Create: `experiments/autoregressive_draft/<diagnostic-run-tag>/`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/audits/2026-08-15-phase1-prompt-to-artifact-coverage.md`

**Interfaces:**
- Produces a checksum-covered focused diagnostic and an evidence-based next
  optimization decision.

- [ ] Sync current source and run the full remote executor suite.
- [ ] Run the focused TP4 batch-4 diagnostic without disturbing GPU-7.
- [ ] Run remote and local independent verifiers.
- [ ] Generate and verify the checksum manifest.
- [ ] Compare all eight repeats and six detail distributions.
- [ ] Select CUDA Graph, authority optimization, metadata optimization, or
  instability investigation using the design decision rule.
- [ ] Record exact evidence and claim boundaries in the bundle README,
  handoff, and audit.
