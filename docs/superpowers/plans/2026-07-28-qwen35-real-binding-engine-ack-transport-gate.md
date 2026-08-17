# Qwen3.5 Real-Binding Engine Acknowledgement Transport Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transport authoritative real-candidate binding rows through the production acknowledgement envelope, executor, pipe, collector, Engine acknowledged call, and Engine all-rank binding validator at TP1 and TP2 without constructing Engine or ModelRunner.

**Architecture:** Freeze and AST-compile two `LLMEngine` methods and one `ModelRunner` dispatch method. Use production acknowledgement dataclasses/executor/collector with real one-way multiprocessing pipes and private rank shells, then aggregate fixed success and failure modes in fresh remote CPU processes.

**Tech Stack:** Python standard library AST/compiler, multiprocessing pipes/processes, existing production acknowledgement module, source-bound SSH orchestration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Use only `sitian@10.232.195.203`.
- Use prerequisite SHA256 `79e140190376a01fb7c07cf80202432dd85791dc6112376a334e13ac9a81048a`.
- Use prerequisite source tree `0d69c3cb59a0bab1a3b19c2846bf2326afff71ca0908e53f7ff7a45c36335785`.
- Freeze `llm_engine.py`, `model_runner.py`, `model_runner_command_ack.py`, and the three exact method hashes from the design.
- Never import or construct `LLMEngine` or `ModelRunner`.
- Never load checkpoint metadata or payloads and never construct model/target/adapter objects.
- Never call scheduler, `LLMEngine.step()`, CUDA, forward, or inference.
- Use real one-way multiprocessing command and acknowledgement pipes for TP2.
- Preserve all evidence and failed runs.
- Do not stage, commit, merge, overwrite, or delete evidence.
- Do not claim latency, throughput, cache, GPU-memory, compression, or quality benefit.

---

### Task 1: Frozen Source and Prerequisite TDD

**Files:**
- Create: `tools/test_qwen35_real_binding_engine_ack_transport_preflight.py`
- Create: `tools/qwen35_real_binding_engine_ack_transport_preflight.py`

**Interfaces:**
- Consumes: authoritative published-binding artifact and frozen production sources.
- Produces: strict prerequisite loader and three frozen callable methods.

- [x] **Step 1: Write prerequisite/source/method RED tests**

  Require the exact artifact SHA/schema/source tree, six unique prior PIDs,
  exact 51-file inheritance, 54-file new closure, exact file/method hashes,
  exact arguments, and zero production module imports.

- [x] **Step 2: Run RED**

  Run the focused test and require the missing preflight module failure.

- [x] **Step 3: Implement strict prerequisite loading**

  Select exact TP1/TP2 success and conflict rows while retaining immutable
  row identity and source closure.

- [x] **Step 4: Implement frozen AST extraction**

  Compile only `call_model_runner_acknowledged`,
  `bind_qwen35_loaded_checkpoint_candidates`, and `dispatch_command` with
  their exact globals.

- [x] **Step 5: Add dependency-light method composition tests**

  Prove TP1 local-only success, TP2 dispatch/collect ordering, completion
  commit, and exact-repeat zero-dispatch.

- [x] **Step 6: Run focused GREEN and Python 3.9 compile**

---

### Task 2: Real Pipe Transport and Failure Semantics

**Files:**
- Modify: `tools/test_qwen35_real_binding_engine_ack_transport_preflight.py`
- Modify: `tools/qwen35_real_binding_engine_ack_transport_preflight.py`

**Interfaces:**
- Consumes: prerequisite rows, production acknowledgement module, and frozen methods.
- Produces: one fresh-attempt worker for each of six modes.

- [x] **Step 1: Write real-pipe success RED tests**

  Require exact envelope, one command send/receive, one `ok` ack, exact
  collector call, ranked rows, completion tuple, child exit, and replay.

- [x] **Step 2: Write four failure-mode RED tests**

  Cover TP1 local inner error, TP2 worker inner error with outer ack `ok`,
  TP2 worker ack exception with outer ack `error`, and worker exit without ack.

- [x] **Step 3: Implement private Engine/runner/worker shells**

  Use one command pipe and one acknowledgement pipe, real fresh child
  process, production executor, and production collector.

- [x] **Step 4: Implement exact transactions**

  Execute the production Engine binder once, validate success or exact
  failure, prove completion commit/unset state, poison behavior, and no child.

- [x] **Step 5: Implement strict row validators**

  Validate exact command IDs, envelope/ack fields, process identities,
  ranked rows, error layers, replay, closure, and no CUDA/forward markers.

- [x] **Step 6: Run focused GREEN and compile**

---

### Task 3: Source-Bound Orchestration and Independent Verifier

**Files:**
- Modify: `tools/test_qwen35_real_binding_engine_ack_transport_preflight.py`
- Modify: `tools/qwen35_real_binding_engine_ack_transport_preflight.py`
- Create: `tools/test_verify_qwen35_real_binding_engine_ack_transport_gate.py`
- Create: `tools/verify_qwen35_real_binding_engine_ack_transport_gate.py`

**Interfaces:**
- Consumes: 54-file closure and one prerequisite.
- Produces: remote runner/finalizer/CLI, two artifacts, and stdlib verifier.

- [x] **Step 1: Write orchestration and verifier RED tests**

  Require six fresh outer PIDs, four unique TP2 child PIDs, fixed ordering,
  separate finalizer, partial rejection, and tamper rejection.

- [x] **Step 2: Implement deterministic staging**

  Stage 54 exact files plus one prerequisite and rehash all frozen methods
  remotely before any attempt.

- [x] **Step 3: Implement workers/finalizer/CLI**

  Run all six modes and atomically publish
  `engine_ack_transport_preflight.json` and `source_manifest.json`.

- [x] **Step 4: Implement standard-library-only verifier**

  Recompute source/method hashes and independently validate every transport,
  completion, failure, poison, liveness, PID, and inventory claim.

- [x] **Step 5: Run static safety audit**

  Require no Engine/ModelRunner import/construction, no checkpoint/model/
  scheduler/step/CUDA/forward/inference path, and exact production ack calls.

- [x] **Step 6: Verify repository hygiene**

  Verify `step()` remains binding-free, hard rejection/schema-v2 unchanged,
  `git diff --check`, and staged files zero.

---

### Task 4: Live Gate, Regression, and Handoff

**Files:**
- Create: `experiments/qwen35_hybrid_state/<run-tag>/engine_ack_transport_preflight.json`
- Create: `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/specs/2026-07-28-qwen35-real-binding-engine-ack-transport-gate-design.md`
- Modify: `docs/superpowers/plans/2026-07-28-qwen35-real-binding-engine-ack-transport-gate.md`

**Interfaces:**
- Consumes: completed production binding gate.
- Produces: authoritative Engine acknowledgement transport evidence and the next shared-memory boundary.

- [x] **Step 1: Run one unique remote gate**

  Preserve failed tags and never overwrite artifacts.

- [x] **Step 2: Independently verify evidence**

  Run the stdlib verifier locally and against the remotely staged source.

- [x] **Step 3: Verify inventory and hashes**

  Require 54 remote source files, one remote input, two remote results, two
  local results, and exact local/remote result SHA equality.

- [x] **Step 4: Run regression matrix**

  Run focused transport/verifier, command ack, live ack wiring, Engine
  all-rank binding, published binding, and adjacent configuration tests.

- [x] **Step 5: Record evidence and claim boundary**

  Record run tag, hashes, PIDs, envelopes, acknowledgements, completion,
  failures, verifier/regression counts, static audit, and absent paths.

- [x] **Step 6: Final verification**

  Re-run focused tests, compile, local/remote validate, independent verifier,
  hashes, audit, `git diff --check`, staged-zero, and complete all checkboxes.
