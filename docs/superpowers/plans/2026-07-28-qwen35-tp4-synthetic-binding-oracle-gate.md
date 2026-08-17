# Qwen3.5 TP4 Synthetic Binding Oracle Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute production TP4 shared-memory acknowledgement transport and the production Engine all-rank binder against an independently reproducible, explicitly synthetic four-rank identity oracle.

**Architecture:** Bind a SHA-locked synthetic oracle artifact to the completed TP4 fan-out prerequisite, freeze six production methods, and execute four fresh remote modes through one shared-memory broadcast and three worker loops. Success commits homogeneous synthetic identity; three rank2 mismatch modes remain transport-successful but fail the production binder without completion.

**Tech Stack:** Python standard library JSON/hash/AST/compiler, `multiprocessing`, POSIX `SharedMemory`, production acknowledgement executor/collector, source-bound SSH orchestration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Use only `sitian@10.232.195.203`.
- Use TP4 prerequisite SHA256 `ec9c07ba903859dbc616dc6c799db4f977284539f9b09cdd85cc57da1a334f8a`.
- Use TP4 prerequisite source tree `ec7b0dee43a06c47b72f8ac14ab26518845f57f070e6c27d394bb4c328644403`.
- Use synthetic oracle SHA256 `1fdc3d64178308d6d26242805433ee31012890f6824a13826075db1e6937431e`.
- Preserve oracle provenance `synthetic-construction-free-oracle`, claim boundary `not-real-checkpoint-binding`, and tensor payload `absent`.
- Never import or construct `LLMEngine` or `ModelRunner`.
- Never read checkpoint metadata or tensor payloads or construct target/adapter/model objects.
- Never call scheduler, `LLMEngine.step()`, CUDA, forward, or inference.
- Never describe synthetic rows as real checkpoint binding evidence.
- Preserve all evidence and failed runs.
- Do not stage, commit, merge, overwrite, or delete evidence.
- Do not claim performance, cache, memory, compression, correctness, or quality benefit.

---

### Task 1: Oracle and Frozen Production Contract

**Files:**
- Create: `tools/test_qwen35_tp4_synthetic_binding_oracle_preflight.py`
- Create: `tools/qwen35_tp4_synthetic_binding_oracle_preflight.py`

**Interfaces:**
- Consumes: immutable TP4 artifact, immutable synthetic oracle, and frozen production sources.
- Produces: independent oracle reconstruction, exact source closure, and six frozen methods.

- [x] **Step 1: Write dual-prerequisite RED tests**

  Require both exact artifact hashes, TP4 source tree, oracle schema/provenance/
  claim boundary/tensor absence, exact four cases, and exact 56-to-57 source
  closure inheritance.

- [x] **Step 2: Run RED**

  Run `python3 tools/test_qwen35_tp4_synthetic_binding_oracle_preflight.py`
  and require failure because the preflight module is absent.

- [x] **Step 3: Implement independent oracle reconstruction**

  Canonicalize the two public descriptors, recompute model/layout and
  alternate hashes, validate exact case rows, and reject any non-synthetic or
  payload-bearing oracle.

- [x] **Step 4: Implement frozen AST extraction**

  Compile `write_shm`, `read_shm`, `loop`, `dispatch_command`,
  `call_model_runner_acknowledged`, and
  `bind_qwen35_loaded_checkpoint_candidates` with exact hashes/signatures.

- [x] **Step 5: Write source/audit RED tests**

  Require 57 sources, no Engine/ModelRunner import/construction, no fixed
  segment name, and zero checkpoint/model/scheduler/step/CUDA/forward paths.

- [x] **Step 6: Implement source closure/static audit and run GREEN**

  Run focused tests and Python 3.9 compile.

---

### Task 2: TP4 Production Binder Transaction

**Files:**
- Modify: `tools/test_qwen35_tp4_synthetic_binding_oracle_preflight.py`
- Modify: `tools/qwen35_tp4_synthetic_binding_oracle_preflight.py`

**Interfaces:**
- Consumes: validated oracle case rows and frozen TP4 transport/binder methods.
- Produces: one fresh production binder attempt for each of four modes.

- [x] **Step 1: Write homogeneous success RED test**

  Require reverse worker completion `(3,2,1)`, ranked collector `(1,2,3)`,
  exact rows, canonical completion tuple, repeat identity, zero repeat binding
  dispatch, two envelopes, and complete cleanup.

- [x] **Step 2: Run success RED**

  Require failure because the transaction worker is absent.

- [x] **Step 3: Implement rank shells and production binder call**

  Reuse only transport primitives, create one segment/three Events/three pipes,
  return exact oracle rows from the public binding method, and invoke frozen
  production acknowledged call plus production binder.

- [x] **Step 4: Implement exact repeat and cleanup**

  Record binding dispatch separately from exit dispatch, prove exact repeat
  does not dispatch, wait for Event clear, send exit, join children, unlink
  once, and prove non-attachability.

- [x] **Step 5: Write three rank2 mismatch RED tests**

  Require all `ok` acknowledgements, healthy collector, ranked results, exact
  field-specific binder error, unset completion, and unchanged other fields.

- [x] **Step 6: Implement mismatch validators and run GREEN**

  Enforce one authorized rank2 field change per case and all per-rank method,
  Event, acknowledgement, completion, child, and shared-memory evidence.

---

### Task 3: Orchestration and Independent Verifier

**Files:**
- Modify: `tools/test_qwen35_tp4_synthetic_binding_oracle_preflight.py`
- Modify: `tools/qwen35_tp4_synthetic_binding_oracle_preflight.py`
- Create: `tools/test_verify_qwen35_tp4_synthetic_binding_oracle_gate.py`
- Create: `tools/verify_qwen35_tp4_synthetic_binding_oracle_gate.py`

**Interfaces:**
- Consumes: exact 57-file closure and two immutable prerequisites.
- Produces: remote run/finalizer/manifest plus independent verifier.

- [x] **Step 1: Write orchestration/verifier RED tests**

  Require four outer/twelve child PIDs, four shared-memory names, separate
  finalizer, partial rejection, verifier independence, and exact inventory.

- [x] **Step 2: Implement deterministic dual-prerequisite staging**

  Stage 57 sources plus TP4 artifact and oracle artifact, rehash all sources,
  methods, descriptors, and artifacts remotely before attempts.

- [x] **Step 3: Implement remote workers/finalizer/CLI**

  Run four fresh modes and atomically publish
  `tp4_synthetic_binding_oracle_preflight.json` and `source_manifest.json`.

- [x] **Step 4: Implement stdlib-only verifier**

  Independently validate every prerequisite, source/method, oracle descriptor,
  transport, ordering, binder, mismatch, completion, and cleanup claim.

- [x] **Step 5: Add directed tamper tests**

  Reject modified oracle provenance and a rank2 mismatch case that changes an
  unauthorized second identity field.

- [x] **Step 6: Run local safety/boundary matrix**

  Run harness/verifier tests, compile, worker hard-rejection AST check,
  schema-v2 `NO_GO` SHA check, `git diff --check`, and staged-zero.

---

### Task 4: Authoritative Remote Evidence and Handoff

**Files:**
- Create: `experiments/qwen35_hybrid_state/<run-tag>/tp4_synthetic_binding_oracle_preflight.json`
- Create: `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/specs/2026-07-28-qwen35-tp4-synthetic-binding-oracle-gate-design.md`
- Modify: `docs/superpowers/plans/2026-07-28-qwen35-tp4-synthetic-binding-oracle-gate.md`

**Interfaces:**
- Consumes: completed TP4 transport gate and immutable synthetic oracle.
- Produces: authoritative TP4 synthetic binder evidence and next real-binding safety boundary.

- [x] **Step 1: Run one unique remote gate**

  Use the exact remote Python and preserve every failed tag.

- [x] **Step 2: Verify locally and remotely**

  Require identical independent verifier check counts over exact staged source
  and both prerequisite artifacts.

- [x] **Step 3: Verify inventory/hashes/resources**

  Require 57 sources, two inputs, two results, exact local/remote hashes, zero
  residual outer/child/finalizer PIDs, and four non-attachable segment names.

- [x] **Step 4: Run adjacent regression matrix**

  Run new tests, TP4 transport tests/verifier, Engine ack tests/verifier,
  command/live ack, all-rank binding, worker boundary, and remote loader config.

- [x] **Step 5: Record evidence and exact synthetic claim boundary**

  Record hashes, descriptors, PIDs, names, ordering, completion/mismatches,
  tests, verifier counts, cleanup, and all absent runtime/performance paths.

- [x] **Step 6: Perform final completion audit**

## Completion Evidence

Authoritative run:

```text
qwen35-tp4-synthetic-binding-20260728-122021
```

Authoritative identities:

```text
source tree:
  e88236ebe4f97ddecf55004e4bbcdb46a677462f183b6724031d85d8648a6de0
result:
  803c8fac331eeee82b90013e0b0872de8f079661b6dd1ba43225fb446006cce4
source manifest:
  643e8d1e24e97ee085f060559999fa1ad1b7608c7c1998c4aaeef9610cc7ccdb
TP4 prerequisite:
  ec9c07ba903859dbc616dc6c799db4f977284539f9b09cdd85cc57da1a334f8a
synthetic oracle:
  1fdc3d64178308d6d26242805433ee31012890f6824a13826075db1e6937431e
```

The standard-library-only verifier passed 720 checks locally and remotely.
Its four tests include oracle-provenance tamper rejection, unauthorized
rank2 second-field tamper rejection, and a re-signed source-tree attack that
adds a production Engine import.

The remote inventory contained exactly 57 source files, two prerequisite
inputs, and two result files. It contained no `__pycache__` or `.pyc` files.
All four outer processes and twelve child processes exited, and all four
shared-memory names were independently non-attachable.

The adjacent local regression matrix passed 69 tests. The remote
torch-dependent manifest-bound loader-configuration suite passed four tests.
Compilation, exact worker hard rejection, schema-v2 canonical `NO_GO` SHA,
`git diff --check`, and staged-zero checks passed.

This plan is complete only for the explicitly synthetic,
`not-real-checkpoint-binding` gate. It does not complete the long-term
performance, correctness, cache, memory, or real-checkpoint objective.

  Map every spec requirement to artifact/test evidence, rerun fresh focused
  verification, close all 24 checkboxes, and keep the long-term goal active.
