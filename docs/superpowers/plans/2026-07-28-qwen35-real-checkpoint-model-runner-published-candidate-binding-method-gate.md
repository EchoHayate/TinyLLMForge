# Qwen3.5 Real-Checkpoint ModelRunner Published-Candidate Binding Method Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the exact production ModelRunner published-candidate, candidate, and owner binding methods on complete approved real Qwen3.5 candidates at TP=1 and TP=2 without importing or constructing ModelRunner.

**Architecture:** Reuse the completed load-and-publish artifact as the immutable source/value prerequisite. Parse the frozen `model_runner.py`, extract and compile the local publication method plus the three-method binding chain, bind those functions onto a minimal private runner shell, execute one successful binding or one injected pre-mutation bridge conflict, and discard the whole private graph.

**Tech Stack:** Python standard library AST/compiler, PyTorch CPU, existing authorized adapter, existing production publication slot, source-bound SSH orchestration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Use only `sitian@10.232.195.203`.
- Use prerequisite SHA256 `d5e6de1ec4a308945897c125eaf7ecff57c44710600ce607db4fd0ae7fb90e18`.
- Use prerequisite source tree `a1bf0161eeedf3c73fb176a0f26ab2156bb3d944096db187a9c83eeb98ae5cc8`.
- Freeze `tinyvllm/engine/model_runner.py` SHA256 `0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849`.
- Freeze publication method SHA256 `37f95954c287d5dd0e8883f299d7049e66dcb3c79806624eb6da3ca7d51a6d4f`.
- Freeze owner binder SHA256 `462e2fefe22e90e60b85c786de6a95e7eaaae31bd9b257025088cd767555ee25`.
- Freeze candidate binder SHA256 `a14e6856ad74eb935116075ee6fe81516c8e212f89914e0fcd55bb39e86d63e0`.
- Freeze outer binder SHA256 `aa178f886d314893593039c5e890239fb954740f059b2d12fc697bd25790fbcd`.
- Use exactly bindings `0..319`.
- Use `max_tensor_bytes=1017118720`.
- Use authorization SHA256 `10a39d6eb918cb5e8d1ccf52a723cdca4db4dffb9fd4ded62b1b766474d4fde4`.
- Build and invoke the existing authorized adapter exactly once per worker.
- Invoke each extracted production method exactly once per worker.
- Never import `tinyvllm.engine.model_runner`.
- Never construct ModelRunner or call `ModelRunner.__init__`.
- Never call `target.take()` or `load_qwen35_fresh_checkpoint_candidate()` directly.
- Never execute Engine, scheduler, `LLMEngine.step()`, CUDA, forward, or inference.
- Never modify production ModelRunner, owner, identity, publication, adapter, factory, loader, worker, or Engine files.
- Clear every selected destination and prove whole-scope collection.
- Preserve all evidence and schema-v2 canonical `NO_GO`.
- Do not stage, commit, merge, overwrite, or delete evidence.
- Do not claim latency, throughput, cache, GPU-memory, compression, or quality benefit.

---

### Task 1: Frozen Four-Method Extraction and Prerequisite TDD

**Files:**
- Create: `tools/test_qwen35_real_checkpoint_model_runner_published_candidate_binding_preflight.py`
- Create: `tools/qwen35_real_checkpoint_model_runner_published_candidate_binding_preflight.py`

**Interfaces:**
- Consumes: load-and-publish artifact and frozen `tinyvllm/engine/model_runner.py`.
- Produces: `load_model_runner_published_binding_prerequisite(...)` and `load_frozen_model_runner_published_binding_methods(...)`.

- [x] **Step 1: Write prerequisite and AST RED tests**

  Require exact prerequisite SHA/schema/source tree, six prior PIDs, 50-file
  inherited source closure, ModelRunner file SHA, all four method source
  hashes, exact arguments, exact three dependency globals, one call through
  each method layer, prevalidated owner mutation ordering, and bounded outer
  exception handling.

- [x] **Step 2: Run RED**

  Run:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.12 \
    tools/test_qwen35_real_checkpoint_model_runner_published_candidate_binding_preflight.py
  ```

  Expected: missing preflight module.

- [x] **Step 3: Implement strict prerequisite loading**

  Validate the immutable artifact and select success value-oracle rows for TP
  `(1,0)`, `(2,0)`, and `(2,1)`. Require the exact 50-file inherited source
  closure.

- [x] **Step 4: Implement exact four-method extraction**

  Parse the frozen file, select exactly one `ModelRunner` class and four named
  methods, recompute every method source SHA, validate globals and structural
  ordering, and compile only those functions with
  `Qwen35HybridModelOwner`, `Qwen35LoadedCheckpointCandidate`, and
  `_bind_qwen35_hybrid_prefix_runtime_identity`.

- [x] **Step 5: Write dependency-light composition RED/GREEN tests**

  Bind all extracted functions onto a minimal shell. Require exact success
  owner/bridge/identity mutation and injected incompatible-bridge bounded
  error with pristine owner/identity state.

- [x] **Step 6: Run focused GREEN and Python 3.9 compile**

  Run focused tests with Python 3.12 and compile both files with
  `/usr/bin/python3 -m py_compile`.

---

### Task 2: Real Candidate Binding Success and Injected Bridge Conflict

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_model_runner_published_candidate_binding_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_model_runner_published_candidate_binding_preflight.py`

**Interfaces:**
- Consumes: approved metadata, authorized adapter, production slot, extracted methods, and prerequisite oracle row.
- Produces: `run_model_runner_published_binding_rank_worker(...)`.

- [x] **Step 1: Write success-row RED tests**

  Require one adapter/provider/publication/outer/candidate/owner call; exact
  bound row; exact owner/bridge/identity pointers; approved fingerprint,
  owner layout fingerprint, `bfloat16`; exact 320/26/aggregate hashes;
  cleanup, collection, and CUDA false.

- [x] **Step 2: Write injected-conflict RED tests**

  Install one incompatible private bridge before binding. Require the exact
  bounded error row, exact candidate publication visibility, unchanged
  injected bridge, empty owner/identity fields, exact values before cleanup,
  cleanup, and collection.

- [x] **Step 3: Implement private graph and runner shell**

  Build metadata, target, authorized adapter, production slot, candidate, and
  minimal shell inside the nested private scope. Bind the three production
  binding functions as shell methods.

- [x] **Step 4: Implement success/failure transactions**

  Publish via the extracted production publication method, invoke the outer
  binding method once, validate success or exact injected rejection, rehash
  values, and clear selected destinations in `finally`.

- [x] **Step 5: Implement row and memory validators**

  Define exact `success` and `injected_bridge_conflict` schemas, including all
  method counts, published visibility, binding visibility, pristine failure
  state, collection, and inherited memory ceilings.

- [x] **Step 6: Run focused GREEN and compile**

  Run all focused tests and Python 3.9 compilation.

---

### Task 3: Source-Bound Orchestration and Static Safety

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_model_runner_published_candidate_binding_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_model_runner_published_candidate_binding_preflight.py`

**Interfaces:**
- Consumes: 50-file prerequisite closure, one prerequisite artifact, and six worker contexts.
- Produces: exact 51-file staging, CLI modes, fresh workers, finalizer, and atomic artifacts.

- [x] **Step 1: Write orchestration RED tests**

  Require 51 unique files, exact 50-file prerequisite inheritance, frozen
  ModelRunner file/four-method SHAs, one prerequisite transfer, six fresh PIDs,
  fixed ordering, separate finalizer, and no result on partial rows.

- [x] **Step 2: Implement deterministic staging**

  Stage the exact source tar and prerequisite, verify all remote hashes, and
  independently recompute all four method hashes remotely before workers.

- [x] **Step 3: Implement workers/finalizer/CLI**

  Launch:

  ```text
  (1,0,success), (1,0,injected_bridge_conflict),
  (2,0,success), (2,0,injected_bridge_conflict),
  (2,1,success), (2,1,injected_bridge_conflict)
  ```

  Atomically publish
  `model_runner_published_candidate_binding_preflight.json` and
  `source_manifest.json`.

- [x] **Step 4: Add strict validation**

  Require approved checkpoint/prerequisite paths, deterministic JSON, Python
  3.9 compatibility, fixed source/method hashes, row ordering, and unique PIDs.

- [x] **Step 5: Run AST safety audit**

  Prove one adapter-builder, one production slot constructor, one invocation
  of each extracted method, zero ModelRunner imports/construction, zero direct
  streamed-loader/target-take/Engine/scheduler/forward/inference calls, and
  only read-only CUDA initialization observations.

- [x] **Step 6: Verify repository hygiene**

  Verify exact worker hard rejection, no production module modifications from
  this plan, `git diff --check`, and staged files zero.

---

### Task 4: Live Gate, Independent Verification, Regression, and Handoff

**Files:**
- Create: `experiments/qwen35_hybrid_state/<run-tag>/model_runner_published_candidate_binding_preflight.json`
- Create: `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/specs/2026-07-28-qwen35-real-checkpoint-model-runner-published-candidate-binding-method-gate-design.md`
- Modify: `docs/superpowers/plans/2026-07-28-qwen35-real-checkpoint-model-runner-published-candidate-binding-method-gate.md`

**Interfaces:**
- Consumes: completed load-and-publish gate and approved checkpoint.
- Produces: authoritative production published-candidate binding evidence and the next Engine acknowledgement boundary.

- [x] **Step 1: Run one unique remote gate**

  Preserve failed tags. If a post-cleanup memory ceiling fails, capture exact
  evidence, add a RED ceiling test, calibrate only to a 256-MiB boundary with
  256 MiB headroom, and rerun under a new tag.

- [x] **Step 2: Independently verify evidence**

  Use a standard-library-only verifier to recompute source/four-method AST
  hashes, compare all value hashes with the prerequisite oracle, and validate
  success binding, injected rejection, cleanup, collection, memory, and six
  unique PIDs.

- [x] **Step 3: Verify inventory and hashes**

  Require:

  ```text
  remote source files: 51
  remote root input artifacts: 1
  remote root result artifacts: 2
  local result artifacts: 2
  local/remote result SHA256: exact equality
  ```

- [x] **Step 4: Run regression matrix**

  Run focused binding, load-and-publish, local/private publication, ownership,
  production slot, atomic candidate binding, runtime identity, ModelRunner
  binding, Engine binding, loader-core, complete transaction, request/
  configuration, factory/loader/metadata/reader/assignment/authorization/
  safety scripts.

- [x] **Step 5: Record exact evidence and claim boundary**

  Record run tag, hashes, four method identities, PIDs, memory, success/error
  rows, owner/bridge/identity evidence, verifier/regression counts, static
  audit, preserved runs, proven boundary, absent paths, and next safe gate.

- [x] **Step 6: Final verification**

  Re-run focused tests, compile, local/remote CLI validation, independent
  verifier, artifact/source/method hashes, AST audit, `git diff --check`, and
  staged-files-zero before completing all checkboxes.
