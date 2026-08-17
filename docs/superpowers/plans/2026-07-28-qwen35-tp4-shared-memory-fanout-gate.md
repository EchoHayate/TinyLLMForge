# Qwen3.5 TP4 Shared-Memory Fan-Out Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove construction-free TP4 production shared-memory fan-out through three real worker loops and three acknowledgement pipes with deterministic ranked collection and exact failure cleanup.

**Architecture:** Freeze five production methods, run one rank0 shell and three fresh rank1/rank2/rank3 loop children against one unique 1 MiB POSIX shared-memory segment, and use three Events plus the production acknowledgement collector. Drive four source-bound remote attempts covering reverse completion, an inner row error, an acknowledgement exception, and `SystemExit` without acknowledgement.

**Tech Stack:** Python standard library AST/compiler, `multiprocessing`, `multiprocessing.shared_memory.SharedMemory`, production acknowledgement dataclasses/executor/collector, source-bound SSH orchestration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Use only `sitian@10.232.195.203`.
- Use prerequisite SHA256 `11f2decd379de668b575cb7f4a0c55874fbefb740d2b4841fb4db3b72ca39c57`.
- Use prerequisite source tree `6cc9672dbd80c211ccd64371573fd8de463b773fc5cc3ae7286ad21c9c664572`.
- Freeze the three production files and five exact method hashes in the design.
- Never import or construct `LLMEngine` or `ModelRunner`.
- Never load checkpoint metadata or payloads and never construct model/target/adapter objects.
- Never call scheduler, `LLMEngine.step()`, CUDA, forward, or inference.
- Use one unique named POSIX shared-memory segment, three real Events, and three real acknowledgement pipes per attempt.
- Never create or unlink the fixed shared-memory name `tinyvllm`.
- Treat identity rows only as a transport oracle, never as real TP4 checkpoint binding.
- Preserve all evidence and failed runs.
- Do not stage, commit, merge, overwrite, or delete evidence.
- Do not claim latency, throughput, cache, GPU-memory, compression, or quality benefit.

---

### Task 1: Frozen Methods and TP4 Fan-Out Contract

**Files:**
- Create: `tools/test_qwen35_tp4_shared_memory_fanout_preflight.py`
- Create: `tools/qwen35_tp4_shared_memory_fanout_preflight.py`

**Interfaces:**
- Consumes: exact TP2 live shared-memory prerequisite and production source files.
- Produces: strict prerequisite loader, five frozen methods, TP4 identity-row contract, and source closure.

- [x] **Step 1: Write prerequisite and source-identity RED tests**

  Require prerequisite SHA
  `11f2decd379de668b575cb7f4a0c55874fbefb740d2b4841fb4db3b72ca39c57`,
  source tree
  `6cc9672dbd80c211ccd64371573fd8de463b773fc5cc3ae7286ad21c9c664572`,
  exact 55-file inheritance, exact 56-file new closure, and all frozen
  file/method hashes and signatures.

- [x] **Step 2: Run RED**

  Run:
  `python3 tools/test_qwen35_tp4_shared_memory_fanout_preflight.py`
  and require failure because the preflight module is absent.

- [x] **Step 3: Implement strict prerequisite loading**

  Parse only the exact TP2 artifact, validate it with the inherited validator,
  require four canonical TP2 rows, and expose an immutable prerequisite object
  with sorted source hashes.

- [x] **Step 4: Implement frozen AST extraction**

  Compile `write_shm`, `read_shm`, `loop`, `dispatch_command`, and
  `call_model_runner_acknowledged` with exact hashes, signatures, and minimal
  globals while importing only `model_runner_command_ack.py`.

- [x] **Step 5: Write identity-row and fan-out validation RED tests**

  Require participants `0..3`, exact nonce/operation, ordered worker ranks,
  rejection of any inner `status="error"` row, and explicit language that the
  rows are not checkpoint bindings.

- [x] **Step 6: Implement the minimal immutable row helpers and run GREEN**

  Add row construction/validation plus exact attempt constants, then rerun the
  focused test and Python 3.9 compilation.

---

### Task 2: Three Production Worker Loops and Failure Semantics

**Files:**
- Modify: `tools/test_qwen35_tp4_shared_memory_fanout_preflight.py`
- Modify: `tools/qwen35_tp4_shared_memory_fanout_preflight.py`

**Interfaces:**
- Consumes: frozen methods and TP4 identity-row validator.
- Produces: one fresh TP4 shared-memory transaction for each fixed mode.

- [x] **Step 1: Write reverse-completion success RED test**

  Require one segment, ranks `(1,2,3)`, acknowledgement send order `(3,2,1)`,
  collector return order `(1,2,3)`, exact participant rows, two envelopes, six
  total worker reads/executions, all child exits zero, and unlink proof.

- [x] **Step 2: Run success RED**

  Run the single success test and require failure because the TP4 worker and
  transaction functions are not implemented.

- [x] **Step 3: Implement parent and three child shells**

  Parent creates one segment, three shared counting Events, three one-way ack
  pipes, and three readiness pipes. Children attach by the same name, inject
  rank-specific delay/behavior, record ack send order, and execute frozen
  production `loop()`.

- [x] **Step 4: Implement acknowledged transaction and safe cleanup**

  Execute frozen `call_model_runner_acknowledged`, wait for all command Events
  to clear before overwriting the shared payload, send one production
  fire-and-forget exit envelope to live workers, join every child, close every
  handle, parent-unlink once, and prove post-unlink attach failure.

- [x] **Step 5: Write three failure-mode RED tests**

  Cover rank2 inner error with all `ok` acks and healthy collector, rank2
  `RuntimeError` with `error` ack and poisoned collector, and rank2
  `SystemExit(9)` without ack with poisoned collector while ranks 1 and 3
  receive cleanup exit.

- [x] **Step 6: Implement failure behaviors/strict row validation and GREEN**

  Validate exact per-mode acknowledgements, inner/outer failure layer,
  completion order, poison state, Event/read/executor counts, child exit codes,
  join state, and shared-memory cleanup; run all focused tests and compile.

---

### Task 3: Source-Bound Orchestration and Independent Verifier

**Files:**
- Modify: `tools/test_qwen35_tp4_shared_memory_fanout_preflight.py`
- Modify: `tools/qwen35_tp4_shared_memory_fanout_preflight.py`
- Create: `tools/test_verify_qwen35_tp4_shared_memory_fanout_gate.py`
- Create: `tools/verify_qwen35_tp4_shared_memory_fanout_gate.py`

**Interfaces:**
- Consumes: exact 56-file closure and one immutable prerequisite artifact.
- Produces: remote staging/workers/finalizer/CLI, two artifacts, and a standard-library-only verifier.

- [x] **Step 1: Write orchestration and verifier RED tests**

  Require four fresh outer PIDs, twelve unique child PIDs, four unique
  shared-memory names, fixed row ordering, separate finalizer, partial-row
  rejection, and verifier independence from TinyLLMForge and gate imports.

- [x] **Step 2: Implement deterministic staging and remote workers**

  Stage exactly 56 files plus one prerequisite, verify every hash remotely,
  launch one fresh outer process per mode, retain failed attempt directories,
  and reject pre-existing output paths.

- [x] **Step 3: Implement aggregate/finalizer/CLI**

  Validate all four rows, atomically publish
  `tp4_shared_memory_fanout_preflight.json` and `source_manifest.json`, support
  local validation, and perform exact remote-to-local copy without overwrite.

- [x] **Step 4: Implement standard-library-only verifier**

  Recompute source/method/prerequisite identities, PID/name uniqueness,
  envelopes, Events, reverse completion and ranked collection, all failure
  layers, child cleanup, unlink/non-attachability, inventory, and hashes.

- [x] **Step 5: Add tamper and static-safety tests**

  Reject a modified collector return order and modified rank2 child exit code.
  Audit zero Engine/ModelRunner import/construction, fixed `tinyvllm`, checkpoint
  loading, scheduler, `step()`, CUDA, forward, and inference calls.

- [x] **Step 6: Run local verifier matrix and repository-boundary checks**

  Run harness/verifier tests, Python 3.9 compile, exact worker hard-rejection
  assertion, schema-v2 `NO_GO` SHA assertion, `git diff --check`, and staged
  file count zero.

---

### Task 4: Remote Gate, Regression, and Handoff

**Files:**
- Create: `experiments/qwen35_hybrid_state/<run-tag>/tp4_shared_memory_fanout_preflight.json`
- Create: `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/specs/2026-07-28-qwen35-tp4-shared-memory-fanout-gate-design.md`
- Modify: `docs/superpowers/plans/2026-07-28-qwen35-tp4-shared-memory-fanout-gate.md`

**Interfaces:**
- Consumes: completed TP2 live shared-memory dispatch gate.
- Produces: authoritative construction-free TP4 fan-out evidence and the next safe runtime boundary.

- [x] **Step 1: Run one unique remote gate**

  Use `sitian@10.232.195.203`, remote Python
  `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`, a unique run
  tag, and preserved attempt/finalizer logs; never overwrite another run.

- [x] **Step 2: Independently verify local and remote evidence**

  Run the standard-library-only verifier against local artifacts and against
  the exact remotely staged source/artifacts, requiring identical check counts.

- [x] **Step 3: Verify inventory, hashes, and resource cleanup**

  Require exactly 56 remote source files, one prerequisite, two results, two
  local results, exact local/remote SHA equality, zero residual outer/child/
  finalizer PIDs, and all four shared-memory names independently non-attachable.

- [x] **Step 4: Run focused and adjacent regression matrices**

  Run the TP4 harness/verifier, TP2 shared-memory harness/verifier, Engine ack
  transport harness/verifier, command-ack, live-ack wiring, all-rank binding,
  published binding, worker boundary, and remote loader-configuration tests.

- [x] **Step 5: Record authoritative evidence and claim boundary**

  Add run tag, hashes, PIDs, names, envelope bytes, Event/read/executor counts,
  send/return orders, acknowledgements, failures, verifier/regression counts,
  static audit, cleanup, and explicitly absent runtime/performance paths to the
  spec and handoff.

- [x] **Step 6: Perform final fresh verification and close all checkboxes**

  Re-run focused tests, compile, local/remote validate, independent verifier,
  hash/inventory/resource checks, boundary audit, `git diff --check`, and
  staged-zero; then mark all 24 steps complete without staging or committing.
