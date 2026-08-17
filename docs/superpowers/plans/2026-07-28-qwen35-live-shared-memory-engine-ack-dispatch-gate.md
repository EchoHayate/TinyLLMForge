# Qwen3.5 Live Shared-Memory Engine Acknowledgement Dispatch Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute authoritative TP2 real-candidate binding rows through the production ModelRunner POSIX shared-memory command path, worker loop, acknowledgement channel, and Engine all-rank binding validator without constructing Engine or ModelRunner.

**Architecture:** Freeze six production methods, run a private rank0 shell in each fresh remote outer process, and run the production `loop()` on one fresh rank1 child attached to a uniquely named real shared-memory segment. Use one real Event, one real acknowledgement pipe, the production acknowledgement collector, and strict independent verification.

**Tech Stack:** Python standard library AST/compiler, `multiprocessing`, `multiprocessing.shared_memory.SharedMemory`, production acknowledgement dataclasses/executor/collector, source-bound SSH orchestration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Use only `sitian@10.232.195.203`.
- Use prerequisite SHA256 `8aeb571c3d56641e747a0d5c5e66314efe6b35b73320cb49e0340c0fe5fd42fb`.
- Use prerequisite source tree `a041ebf7653e141dd96ebe31143ba00e5634c61c1a4bec68f17e7a7c6bba5cc8`.
- Freeze the three production files and six exact method hashes in the design.
- Never import or construct `LLMEngine` or `ModelRunner`.
- Never load checkpoint metadata or payloads and never construct model/target/adapter objects.
- Never call scheduler, `LLMEngine.step()`, CUDA, forward, or inference.
- Use one unique named POSIX shared-memory segment, one real Event, and one real acknowledgement pipe per attempt.
- Never create or unlink the fixed shared-memory name `tinyvllm`.
- Preserve all evidence and failed runs.
- Do not stage, commit, merge, overwrite, or delete evidence.
- Do not claim latency, throughput, cache, GPU-memory, compression, or quality benefit.

---

### Task 1: Frozen Methods and Shared-Memory Codec TDD

**Files:**
- Create: `tools/test_qwen35_live_shared_memory_engine_ack_dispatch_preflight.py`
- Create: `tools/qwen35_live_shared_memory_engine_ack_dispatch_preflight.py`

**Interfaces:**
- Consumes: authoritative Engine acknowledgement artifact and frozen production sources.
- Produces: strict prerequisite loader and six frozen callable methods.

- [x] **Step 1: Write prerequisite/source/method RED tests**

  Require the exact prerequisite SHA/schema/source tree, exact 54-file
  inheritance, 55-file new closure, exact file/method hashes and signatures,
  and zero production Engine/ModelRunner imports.

- [x] **Step 2: Run RED**

  Run the focused test and require failure because the preflight module is
  missing.

- [x] **Step 3: Implement strict prerequisite loading**

  Select exact TP2 rank0/rank1 success and rank1 conflict rows while retaining
  immutable transport identities and source closure.

- [x] **Step 4: Implement frozen AST extraction**

  Compile only `write_shm`, `read_shm`, `loop`, `dispatch_command`,
  `call_model_runner_acknowledged`, and
  `bind_qwen35_loaded_checkpoint_candidates` with exact globals.

- [x] **Step 5: Write real SharedMemory codec RED tests**

  Require one unique named segment, exact envelope serialization, exact byte
  count, real Event set/wait/clear, and post-unlink attach failure.

- [x] **Step 6: Implement codec fixture and run focused GREEN/compile**

---

### Task 2: Production Worker Loop and Failure Semantics

**Files:**
- Modify: `tools/test_qwen35_live_shared_memory_engine_ack_dispatch_preflight.py`
- Modify: `tools/qwen35_live_shared_memory_engine_ack_dispatch_preflight.py`

**Interfaces:**
- Consumes: frozen methods, prerequisite rows, production acknowledgement module.
- Produces: one fresh TP2 transaction for each of four fixed modes.

- [x] **Step 1: Write TP2 success RED test**

  Require dispatch/write/read/loop/executor/collector counts, exact envelope,
  exact rows, completion commit, replay zero-dispatch, exit envelope, child
  join, and shared-memory unlink.

- [x] **Step 2: Write three failure-mode RED tests**

  Cover worker inner binding error with `ok` ack, worker exception with `error`
  ack and poison, and `SystemExit` without ack with worker-death/EOF poison.

- [x] **Step 3: Implement rank0 and rank1 shells**

  Parent creates the segment/Event/ack pipe; child attaches by unique name,
  reports ready, and runs the frozen production `loop()`.

- [x] **Step 4: Implement exact transaction and cleanup**

  Run the Engine binder, send fire-and-forget exit when the child remains
  alive, join, close, unlink, and prove the segment is no longer attachable.

- [x] **Step 5: Implement strict row validators**

  Validate exact command IDs, envelopes, payload bytes, Event/method counts,
  ack/error layers, completion, poison, PIDs, exit, join, and unlink.

- [x] **Step 6: Run focused GREEN and Python 3.9 compile**

---

### Task 3: Source-Bound Orchestration and Independent Verifier

**Files:**
- Modify: `tools/test_qwen35_live_shared_memory_engine_ack_dispatch_preflight.py`
- Modify: `tools/qwen35_live_shared_memory_engine_ack_dispatch_preflight.py`
- Create: `tools/test_verify_qwen35_live_shared_memory_engine_ack_dispatch_gate.py`
- Create: `tools/verify_qwen35_live_shared_memory_engine_ack_dispatch_gate.py`

**Interfaces:**
- Consumes: exact 55-file closure and one prerequisite.
- Produces: remote runner/finalizer/CLI, two artifacts, and stdlib verifier.

- [x] **Step 1: Write orchestration and verifier RED tests**

  Require four fresh outer PIDs, four unique child PIDs, four unique shared
  memory names, fixed ordering, separate finalizer, partial rejection, and
  two tamper rejections.

- [x] **Step 2: Implement deterministic staging**

  Stage 55 exact files plus one prerequisite and rehash all six frozen methods
  remotely before any attempt.

- [x] **Step 3: Implement workers/finalizer/CLI**

  Run all four modes and atomically publish
  `live_shared_memory_engine_ack_dispatch_preflight.json` and
  `source_manifest.json`.

- [x] **Step 4: Implement standard-library-only verifier**

  Independently validate every source, method, shared-memory, Event, envelope,
  acknowledgement, completion, failure, poison, PID, cleanup, and inventory
  claim.

- [x] **Step 5: Run static safety audit**

  Require no Engine/ModelRunner import/construction, no fixed `tinyvllm`
  segment, no checkpoint/model/scheduler/step/CUDA/forward/inference path,
  and exact production shared-memory/ack call sites.

- [x] **Step 6: Verify repository hygiene**

  Verify `step()` remains binding-free, worker hard rejection/schema-v2
  unchanged, `git diff --check`, and staged files zero.

---

### Task 4: Live Remote Gate, Regression, and Handoff

**Files:**
- Create: `experiments/qwen35_hybrid_state/<run-tag>/live_shared_memory_engine_ack_dispatch_preflight.json`
- Create: `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/specs/2026-07-28-qwen35-live-shared-memory-engine-ack-dispatch-gate-design.md`
- Modify: `docs/superpowers/plans/2026-07-28-qwen35-live-shared-memory-engine-ack-dispatch-gate.md`

**Interfaces:**
- Consumes: completed private-pipe Engine acknowledgement transport gate.
- Produces: authoritative live shared-memory Engine dispatch evidence and the next TP4 fan-out boundary.

- [x] **Step 1: Run one unique remote gate**

  Preserve every failed tag and never overwrite artifacts.

- [x] **Step 2: Independently verify evidence**

  Run the stdlib verifier locally and against the remotely staged source.

- [x] **Step 3: Verify inventory, hashes, and resource cleanup**

  Require 55 remote source files, one remote input, two remote results, two
  local results, exact local/remote SHA equality, zero residual workers, and
  zero attachable shared-memory names.

- [x] **Step 4: Run regression matrix**

  Run focused shared-memory/verifier, previous Engine ack gate, command ack,
  live ack wiring, Engine all-rank binding, published binding, and adjacent
  loader configuration tests.

- [x] **Step 5: Record evidence and claim boundary**

  Record run tag, hashes, PIDs, shared-memory names, envelopes, Event/method
  counts, acknowledgements, completion, failures, verifier/regression counts,
  static audit, cleanup, and absent paths.

- [x] **Step 6: Final verification**

  Re-run focused tests, compile, local/remote validate, independent verifier,
  hashes, resource cleanup, audit, `git diff --check`, staged-zero, and
  complete all 24 checkboxes.
