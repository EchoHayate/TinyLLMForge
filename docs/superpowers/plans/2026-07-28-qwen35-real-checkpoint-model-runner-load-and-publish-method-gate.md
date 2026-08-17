# Qwen3.5 Real-Checkpoint ModelRunner Load-and-Publish Method Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the exact production `ModelRunner.load_and_publish_qwen35_checkpoint_candidate(request)` method on complete approved real Qwen3.5 candidates at TP=1 and TP=2 without importing or constructing ModelRunner.

**Architecture:** Reuse the completed publication-method artifact as the source, cleanup, and method-publication prerequisite and the complete-checkpoint artifact as the value oracle. Parse the frozen `model_runner.py`, extract and compile only the exact load-and-publish method with its two production globals, invoke it on a minimal private runner shell with the existing authorized real loader, and validate success or injected publication rejection before discarding the whole private graph.

**Tech Stack:** Python standard library AST/compiler, PyTorch CPU, existing authorized adapter, existing production publication slot, source-bound SSH orchestration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Use only `sitian@10.232.195.203`.
- Use complete prerequisite SHA256 `7ed84bd1e4e894e9ea990ff2dc631db849f805f4468a9b266bd308d690acb176`.
- Use publication-method prerequisite SHA256 `f8f78ae574991eb3f16aed57b4275cf76a409fa553e01597f5179c41eb158b15`.
- Freeze `tinyvllm/engine/model_runner.py` SHA256 `0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849`.
- Freeze method source SHA256 `9134c5bad8c4127714e07ffd8af56209c247a746e9f0d0ceceb60227c1358612`.
- Use exactly bindings `0..319`.
- Use `max_tensor_bytes=1017118720`.
- Use authorization SHA256 `10a39d6eb918cb5e8d1ccf52a723cdca4db4dffb9fd4ded62b1b766474d4fde4`.
- Build and invoke the existing authorized adapter exactly once per worker.
- Extract and invoke the production load-and-publish method exactly once per worker.
- Never import `tinyvllm.engine.model_runner`.
- Never construct ModelRunner or call `ModelRunner.__init__`.
- Never call `target.take()` or `load_qwen35_fresh_checkpoint_candidate()` directly.
- Never execute binding, Engine, scheduler, `LLMEngine.step()`, CUDA, forward, or inference.
- Never modify production ModelRunner, publication, adapter, factory, loader, worker, or Engine files.
- Clear every selected destination and prove whole-scope collection.
- Preserve all evidence and schema-v2 canonical `NO_GO`.
- Do not stage, commit, merge, overwrite, or delete evidence.
- Do not claim latency, throughput, cache, GPU-memory, compression, or quality benefit.

---

### Task 1: Frozen Method Extraction and Prerequisite TDD

**Files:**
- Create: `tools/test_qwen35_real_checkpoint_model_runner_load_and_publish_preflight.py`
- Create: `tools/qwen35_real_checkpoint_model_runner_load_and_publish_preflight.py`

**Interfaces:**
- Consumes: complete artifact, publication-method artifact, and frozen `tinyvllm/engine/model_runner.py`.
- Produces: `load_model_runner_load_publish_prerequisites(...)` and `load_frozen_model_runner_load_publish_method(...)`.

- [x] **Step 1: Write prerequisite and AST RED tests**

  Require exact prerequisite SHA/schema/source trees, six prior PIDs, 49-file
  inherited source closure, ModelRunner file SHA, load-and-publish method
  source SHA, exact arguments, exact two dependency globals, one loader call,
  one slot publish, and completion writes after publication.

- [x] **Step 2: Run RED**

  Run:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.12 \
    tools/test_qwen35_real_checkpoint_model_runner_load_and_publish_preflight.py
  ```

  Expected: missing preflight module.

- [x] **Step 3: Implement strict prerequisites**

  Validate both immutable artifacts and select complete/publication success
  rows for TP `(1,0)`, `(2,0)`, `(2,1)`. Require exact 49-file inherited
  source closure.

- [x] **Step 4: Implement exact method extraction**

  Parse the frozen file, select exactly one `ModelRunner` class and one named
  method, recompute method source SHA, validate dependencies and structural
  ordering, compile only that function with
  `Qwen35LoadedCheckpointCandidate` and
  `validate_qwen35_checkpoint_candidate_load_request`, and reject drift.

- [x] **Step 5: Write dependency-light method-call RED/GREEN tests**

  Use exact fake request/candidate types and a minimal shell. Require success
  publication plus completion commit and proxy rejection plus bounded error
  row with pristine completion state.

- [x] **Step 6: Run focused GREEN and Python 3.9 compile**

  Run focused tests with Python 3.12 and compile both files with
  `/usr/bin/python3 -m py_compile`.

---

### Task 2: Real Candidate Success and Injected Publication Failure

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_model_runner_load_and_publish_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_model_runner_load_and_publish_preflight.py`

**Interfaces:**
- Consumes: approved metadata, exact authorized adapter, exact request, exact production slot, extracted method, and complete oracle row.
- Produces: `run_model_runner_load_publish_rank_worker(...)`.

- [x] **Step 1: Write success-row RED tests**

  Require one method, adapter, provider, and production publish call; exact
  published participant row; exact slot candidate/owner/fingerprint; exact
  completion request/configuration; exact 320/26/aggregate hashes; cleanup,
  collection, and CUDA false.

- [x] **Step 2: Write injected-publication-failure RED tests**

  Use a proxy slot that records one exact candidate then raises
  `RuntimeError("injected ModelRunner load-and-publish failure")` before
  production delegation. Require exact bounded error row, production slot
  empty, completion state `None`, exact values before cleanup, cleanup, and
  collection.

- [x] **Step 3: Implement private graph, request, and runner shell**

  Build metadata, target, authorized adapter, exact production request,
  production slot, and minimal shell inside the nested private scope.

- [x] **Step 4: Implement success/failure transactions**

  Invoke the extracted method once, validate returned row and success
  publication/completion or exact injected rejection, rehash values, and clear
  selected destinations in `finally`.

- [x] **Step 5: Implement row and memory validators**

  Define exact success and `injected_publication_failure` schemas, including
  method/adapter/provider/proxy/production counts, completion state,
  visibility, collection, and memory ceilings.

- [x] **Step 6: Run focused GREEN and compile**

  Run all focused tests and Python 3.9 compilation.

---

### Task 3: Source-Bound Orchestration and Static Safety

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_model_runner_load_and_publish_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_model_runner_load_and_publish_preflight.py`

**Interfaces:**
- Consumes: 49-file publication-method closure, new preflight, two prerequisites, and six worker contexts.
- Produces: exact 50-file staging, CLI modes, fresh workers, finalizer, and atomic artifacts.

- [x] **Step 1: Write orchestration RED tests**

  Require 50 unique files, exact 49-file prerequisite inheritance, frozen
  ModelRunner file/method SHA, two prerequisite transfers, six fresh PIDs,
  fixed ordering, separate finalizer, and no result on partial rows.

- [x] **Step 2: Implement deterministic staging**

  Stage the exact source tar and prerequisites, verify all remote hashes, and
  independently recompute the load-and-publish method SHA remotely before
  workers.

- [x] **Step 3: Implement workers/finalizer/CLI**

  Launch:

  ```text
  (1,0,success), (1,0,injected_publication_failure),
  (2,0,success), (2,0,injected_publication_failure),
  (2,1,success), (2,1,injected_publication_failure)
  ```

  Atomically publish `model_runner_load_and_publish_preflight.json` and
  `source_manifest.json`.

- [x] **Step 4: Add strict validation**

  Require approved checkpoint/prerequisite paths, deterministic JSON, Python
  3.9 compatibility, fixed source/method hashes, row ordering, and unique PIDs.

- [x] **Step 5: Run AST safety audit**

  Prove one adapter-builder call, one production slot constructor, one
  extracted-method invocation, zero ModelRunner imports/construction, zero
  direct streamed-loader/target-take/binding/Engine/scheduler/forward/
  inference calls, and only read-only CUDA initialization observations.

- [x] **Step 6: Verify repository hygiene**

  Verify exact worker hard rejection, no production module modifications from
  this plan, `git diff --check`, and staged files zero.

---

### Task 4: Live Gate, Independent Verification, Regression, and Handoff

**Files:**
- Create: `experiments/qwen35_hybrid_state/<run-tag>/model_runner_load_and_publish_preflight.json`
- Create: `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/specs/2026-07-28-qwen35-real-checkpoint-model-runner-load-and-publish-method-gate-design.md`
- Modify: `docs/superpowers/plans/2026-07-28-qwen35-real-checkpoint-model-runner-load-and-publish-method-gate.md`

**Interfaces:**
- Consumes: completed local gate, approved checkpoint, and two immutable prerequisites.
- Produces: authoritative production load-and-publish evidence and the next published-candidate binding boundary.

- [x] **Step 1: Run one unique remote gate**

  Preserve failed tags. If a post-cleanup memory ceiling fails, capture exact
  evidence, add a RED ceiling test, calibrate only to a 256-MiB boundary with
  256 MiB headroom, and rerun under a new tag.

- [x] **Step 2: Independently verify evidence**

  Use a standard-library-only verifier to recompute source/method AST hashes,
  compare all value hashes with the complete oracle, and validate success,
  injected rejection, completion state, cleanup, collection, memory, and six
  unique PIDs.

- [x] **Step 3: Verify inventory and hashes**

  Require:

  ```text
  remote source files: 50
  remote root input artifacts: 2
  remote root result artifacts: 2
  local result artifacts: 2
  local/remote result SHA256: exact equality
  ```

- [x] **Step 4: Run regression matrix**

  Run focused load-and-publish, publication-method, private publication,
  ownership, production slot, ModelRunner loader/binding, authorized loader,
  Engine binding, loader-core, complete transaction, request/configuration,
  factory/loader/metadata/reader/assignment/authorization/safety scripts.

- [x] **Step 5: Record exact evidence and claim boundary**

  Record run tag, hashes, method source identity, PIDs, memory, success/failure
  participant rows, publication/completion evidence, verifier/regression
  counts, static audit, preserved runs, proven boundary, absent paths, and
  next safe gate.

- [x] **Step 6: Final verification**

  Re-run focused tests, compile, local/remote CLI validation, independent
  verifier, artifact/source/method hashes, AST audit, `git diff --check`, and
  staged-files-zero before completing all checkboxes.
