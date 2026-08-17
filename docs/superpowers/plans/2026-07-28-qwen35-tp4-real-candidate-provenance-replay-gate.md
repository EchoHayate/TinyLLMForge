# Qwen3.5 TP4 Real-Candidate Provenance Replay Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Produce four serial, source-bound real TP4 candidate rows and replay them through the production TP4 acknowledgement transport and all-rank binder without concurrent candidate retention.

**Architecture:** Extend the proven published-candidate binding harness to TP4 ranks 0..3 in separate sequential processes, atomically finalize a real provenance oracle, then feed only its immutable bound rows into the proven TP4 shared-memory transport/binder harness. Independent verification covers both real payload provenance and replay transport while preserving a strict non-live-concurrent claim boundary.

**Tech Stack:** Python 3.9+ standard library, PyTorch CPU, safetensors bounded streaming, `multiprocessing`, POSIX `SharedMemory`, AST extraction/compilation, SSH source-bound orchestration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Use only `sitian@10.232.195.203`.
- Preserve all local and remote evidence, including failed runs.
- Do not stage, commit, merge, create a PR, overwrite evidence, or delete evidence.
- Use provenance `real-checkpoint-derived-serial-rank-replay`.
- Use claim boundary `not-live-concurrent-tp4-candidate-binding`.
- Never import or construct production `LLMEngine` or `ModelRunner`.
- Never call scheduler, `LLMEngine.step()`, CUDA operations, forward, or inference.
- Never run more than one real-candidate producer worker at a time.
- Preserve the exact real worker hard rejection.
- Preserve schema-v2 canonical `NO_GO`.
- Do not claim correctness, quality, latency, throughput, cache, memory, or compression benefit.

---

### Task 1: TP4 Real Producer Contract

**Files:**
- Create: `tools/test_qwen35_tp4_real_candidate_provenance_replay_preflight.py`
- Create: `tools/qwen35_tp4_real_candidate_provenance_replay_preflight.py`

**Interfaces:**
- Consumes: three immutable prerequisite artifacts, approved checkpoint identities, and frozen production methods.
- Produces: strict prerequisite loader, 58-file closure, TP4 producer configuration, and frozen method map.

- [x] **Step 1: Write prerequisite and closure RED tests**

  Require exact prerequisite SHA/source trees, approved manifest/shard/
  authorization values, exact 57-to-58 source inheritance, TP ranks `0..3`,
  provenance, claim boundary, and fixed per-process memory ceilings.

- [x] **Step 2: Run RED**

  Run:

  ```bash
  python3 tools/test_qwen35_tp4_real_candidate_provenance_replay_preflight.py
  ```

  Require failure because the preflight module is absent.

- [x] **Step 3: Implement strict prerequisite loading**

  Load and validate:

  ```text
  d5e6de1...  ModelRunner load-and-publish
  79e14019...  published-candidate binding
  803c8fac...  TP4 synthetic transport/binder
  ```

  Reject any row/source/schema/hash drift and expose immutable prerequisite
  mappings.

- [x] **Step 4: Implement frozen AST extraction**

  Extract, signature-check, hash-check, compile, and return:

  ```text
  load_and_publish_qwen35_checkpoint_candidate
  bind_published_qwen35_loaded_checkpoint_candidate
  write_shm
  read_shm
  loop
  dispatch_command
  call_model_runner_acknowledged
  bind_qwen35_loaded_checkpoint_candidates
  ```

- [x] **Step 5: Write static safety RED tests**

  Require no Engine/ModelRunner import/construction, no scheduler/step/CUDA
  operation/forward/inference, one authorized loader-builder site, no fixed
  `tinyvllm` segment name, and explicit serial producer orchestration.

- [x] **Step 6: Implement source closure/audit and run GREEN**

  Run the focused suite and Python 3.9 compile. Require 58 source files and
  exact inherited hashes for the first 57.

---

### Task 2: Serial TP4 Real Candidate Producers

**Files:**
- Modify: `tools/test_qwen35_tp4_real_candidate_provenance_replay_preflight.py`
- Modify: `tools/qwen35_tp4_real_candidate_provenance_replay_preflight.py`

**Interfaces:**
- Consumes: approved checkpoint and TP4 `(size=4, rank=0..3)`.
- Produces: one complete real producer row per fresh sequential process.

- [x] **Step 1: Write rank0 producer RED test**

  Require one authorized loader call, one target provider call, one production
  load-and-publish call, one production bind call, 320 binding hashes, 26
  phase hashes, aggregate hash, `bfloat16`, TP4 layout fingerprint, exact
  participant row, memory points, cleanup, collection, CUDA false, and zero
  forward calls.

- [x] **Step 2: Run producer RED**

  Require failure because the TP4 producer worker is absent.

- [x] **Step 3: Implement TP4 target/load/publish/bind worker**

  Reuse the existing approved metadata, loader configuration, private runner
  shell, publication slot, binding diagnostics, clear logic, weakrefs, and
  memory reader. Parameterize only TP size/rank and preserve production method
  invocation order.

- [x] **Step 4: Implement exact TP4 payload oracle checks**

  Record all rank-specific binding/phase/aggregate hashes and validate shape,
  assignment, alias, loader-stat, model fingerprint, layout, dtype, selected
  destination, and pool invariants without comparing against TP2 payload
  hashes.

- [x] **Step 5: Write ranks1-3 and serial-order RED tests**

  Require four unique producer PIDs, rank/participant equality, one live
  producer at a time, previous PID absent before the next starts, homogeneous
  model/layout/dtype, and no private-object escape.

- [x] **Step 6: Implement serial coordinator and run GREEN**

  Run ranks `0,1,2,3` strictly sequentially. Reject overlap, ceiling breach,
  incomplete rank set, heterogeneous identity, or collection failure.

---

### Task 3: Provenance Oracle and TP4 Replay

**Files:**
- Modify: `tools/test_qwen35_tp4_real_candidate_provenance_replay_preflight.py`
- Modify: `tools/qwen35_tp4_real_candidate_provenance_replay_preflight.py`

**Interfaces:**
- Consumes: four validated producer rows.
- Produces: immutable provenance oracle plus four production TP4 replay rows.

- [x] **Step 1: Write provenance finalizer RED test**

  Require atomic `tp4_real_candidate_provenance_oracle.json`, exact producer
  ordering, exact payload hashes, producer-exited proof, provenance, claim
  boundary, and rejection of partial or overlapping producer sets.

- [x] **Step 2: Implement immutable oracle finalization**

  Canonicalize and hash the four producer rows only after all producer PIDs are
  absent. Reject any post-validation mutation.

- [x] **Step 3: Write TP4 replay success RED test**

  Require one fresh segment, three Events/pipes/worker loops, send order
  `(3,2,1)`, collector order `(1,2,3)`, exact real-derived rows, committed
  configuration, exact-repeat zero binding dispatch, exit, join, unlink, and
  non-attachability.

- [x] **Step 4: Implement production transport/binder replay**

  Reuse only the immutable rows from the oracle. Do not retain, reload, or
  reconstruct candidates in replay processes.

- [x] **Step 5: Write three directed mismatch RED tests**

  Require rank2-only model/layout/dtype mutation, unchanged producer payload
  evidence, all `ok` acknowledgements, healthy collector, exact binder error,
  unset completion, and complete cleanup.

- [x] **Step 6: Implement mismatch replay and run GREEN**

  Enforce one authorized identity change per negative mode and reject any
  second field or payload-evidence mutation.

---

### Task 4: Orchestration, Verifier, and Remote Evidence

**Files:**
- Modify: `tools/test_qwen35_tp4_real_candidate_provenance_replay_preflight.py`
- Modify: `tools/qwen35_tp4_real_candidate_provenance_replay_preflight.py`
- Create: `tools/test_verify_qwen35_tp4_real_candidate_provenance_replay_gate.py`
- Create: `tools/verify_qwen35_tp4_real_candidate_provenance_replay_gate.py`
- Create: `experiments/qwen35_hybrid_state/<run-tag>/tp4_real_candidate_provenance_oracle.json`
- Create: `experiments/qwen35_hybrid_state/<run-tag>/tp4_real_candidate_provenance_replay_preflight.json`
- Create: `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `docs/superpowers/specs/2026-07-28-qwen35-tp4-real-candidate-provenance-replay-gate-design.md`
- Modify: `docs/superpowers/plans/2026-07-28-qwen35-tp4-real-candidate-provenance-replay-gate.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: exact 58-file closure, three immutable prerequisites, and approved checkpoint.
- Produces: authoritative remote evidence and independent verification.

- [x] **Step 1: Write orchestration/verifier RED tests**

  Require deterministic 58-file tar, three prerequisite transfers, four
  sequential producers, separate oracle/result finalizers, four replay outer
  and twelve child PIDs, exact inventory, partial rejection, and atomic local
  publication.

- [x] **Step 2: Implement source-bound remote CLI**

  Add `run`, `internal-producer-worker`, `internal-finalize-oracle`,
  `internal-replay-worker`, `internal-finalize-result`, and `validate` modes.
  Rehash sources, methods, prerequisites, and approved checkpoint identities
  remotely before starting producers.

- [x] **Step 3: Implement stdlib-only independent verifier**

  Independently check all prerequisite/source/method identities, TP4 payload
  provenance, producer ordering/cleanup/memory, oracle immutability, replay
  transport/binder/mismatch/cleanup, inventory, and local/remote hashes.

- [x] **Step 4: Add directed tamper tests**

  Reject rank/participant mismatch, re-signed production import, unauthorized
  second replay field mutation, and producer/replay PID overlap or ordering
  violation.

- [x] **Step 5: Run authoritative remote gate and regressions**

  Use one unique tag and exact remote Python. Preserve every failed tag.
  Require identical local/remote verifier counts, 58 sources, three inputs,
  one oracle, two results, zero residual PIDs, non-attachable segments, focused
  tests, adjacent real-loader/binding/TP4/ack regressions, remote loader config,
  compile, worker rejection, schema-v2 `NO_GO`, diff check, and staged zero.

- [x] **Step 6: Record evidence and complete audit**

  Record hashes, producer/replay PIDs, memory, payload hashes, identities,
  ordering, mismatch semantics, cleanup, test/verifier counts, exact claim
  boundary, and next live-concurrent TP4 gate. Map every spec requirement to
  evidence and close all 24 checkboxes while keeping the long-term performance
  goal active.

## Completion Evidence

Authoritative pristine run:

```text
qwen35-tp4-real-candidate-replay-20260728-145713
```

Final identities:

```text
source tree:
  42dddc0eac0a6db6041d5abb71df34db4d5e7c99d3b74d69f94598a2f24eb137
oracle:
  d750d664219378c234a2127b708ec191feb9b2c9f1f2902c47d0ad5dc152d3ef
result:
  dc25e1cc72701d994745210022ddbd6bc603054a5acf842400aa48a3159e88e4
manifest:
  f6c1c8846fc478bd341c720665b1230abe896db672434aee8c314764def22ead
```

The standard-library-only verifier passed 2206 checks against the local
three-file publication and 2211 checks against the full remote inventory.
Its five focused tests cover the pristine artifact plus participant mismatch,
unauthorized second-field mutation, PID overlap, and a fully re-signed
production Engine import. The verifier was uploaded to a run-external
directory and the authoritative remote `source/` tree remained untouched.

This gate retains the exact provenance
`real-checkpoint-derived-serial-rank-replay` and claim boundary
`not-live-concurrent-tp4-candidate-binding`. The long-term performance goal
remains active; no latency, throughput, accuracy, quality, cache, memory, or
compression improvement is claimed here.
