# Qwen3.5 Private Candidate Ownership Transfer Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the existing authorized prepared-target adapter on the complete approved real Qwen3.5 checkpoint, prove one-shot private candidate ownership transfer at TP=1 and TP=2, and clear/discard private state after success and injected post-transfer failure.

**Architecture:** Reuse the complete-checkpoint artifact as the immutable value oracle and the completed tiled loader-core artifact as the source/target prerequisite. Build a fresh CPU target per process, call the existing authorized adapter exactly once, validate the exact streamed loaded candidate or injected partial failure, and clear all selected destinations in a fail-closed `finally` block without publication or runtime installation.

**Tech Stack:** Python standard library, PyTorch CPU, safetensors streamed loader, existing Qwen3.5 prepared-target/adapter/request modules, source-bound SSH orchestration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Use only `sitian@10.232.195.203`.
- Use complete prerequisite run `qwen35-complete-checkpoint-20260728-065128`.
- Use loader-core prerequisite run `qwen35-tiled-loader-core-20260728-075700`.
- Use exactly bindings `0..319`.
- Use request `max_tensor_bytes=1017118720`.
- Use authorization SHA256 `10a39d6eb918cb5e8d1ccf52a723cdca4db4dffb9fd4ded62b1b766474d4fde4`.
- Call the existing authorized adapter exactly once per worker.
- Never call `target.take()` or `load_qwen35_fresh_checkpoint_candidate()` directly from the preflight.
- Never install or publish a candidate, call ModelRunner, Engine, scheduler, `LLMEngine.step()`, forward, CUDA, or inference.
- Clear and verify every selected destination after success or failure; preserve unbound rotary buffers and pool state exactly.
- Preserve all failed/superseded evidence and schema-v2 canonical `NO_GO`.
- Do not modify production adapter/factory/streamed-loader/worker/runtime files.
- Do not stage, commit, merge, overwrite, or delete evidence.
- Do not claim production latency, throughput, cache, GPU-memory, compression, or quality benefit.

---

### Task 1: Prerequisite Oracle and Private Cleanup TDD

**Files:**
- Create: `tools/test_qwen35_real_checkpoint_private_candidate_ownership_preflight.py`
- Create: `tools/qwen35_real_checkpoint_private_candidate_ownership_preflight.py`

**Interfaces:**
- Consumes:
  `complete_checkpoint_transaction_preflight.json`,
  `tiled_loader_core_preflight.json`, and one fresh prepared target.
- Produces:
  `load_private_ownership_prerequisites(...)`,
  `validate_private_loaded_candidate(...)`, and
  `clear_private_candidate_target(...)`.

- [x] **Step 1: Write exact prerequisite RED tests**

  Add tests that require:

  ```python
  complete_sha256 = (
      "7ed84bd1e4e894e9ea990ff2dc631db849f805f4468a9b266bd308d690acb176"
  )
  loader_core_sha256 = (
      "58df3dfa9fec11d1fd079c9473766413232bd3f928f537ac87e047e13ef65aae"
  )
  rows = ((1, 0), (2, 0), (2, 1))
  ```

  Require exact schemas, source trees, three unique prerequisite PIDs, 320
  binding results, 26 phase results, 24 alias groups, loader-core cleanup,
  and CUDA false.

- [x] **Step 2: Run RED**

  Run:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.12 \
    tools/test_qwen35_real_checkpoint_private_candidate_ownership_preflight.py
  ```

  Expected: module import failure because the preflight does not exist.

- [x] **Step 3: Implement strict prerequisite loading**

  Implement exact file SHA256 validation, schema/row validation, TP row
  selection, and equality between the current frozen 44-file source hashes
  and the loader-core artifact's `source_file_sha256`.

- [x] **Step 4: Write success cleanup RED tests**

  Build a compact exact target/candidate fixture and require:

  ```text
  target._consumed: false -> true
  provider calls: 1
  adapter calls: 1
  candidate exact type and identity: true
  selected unique destinations: reverse-cleared to zero
  unbound rotary values: unchanged
  pool snapshot: unchanged
  ```

- [x] **Step 5: Implement candidate validation and cleanup**

  Reuse the loader-core helper's exact destination-view, tensor-byte, alias,
  pool-snapshot, and zero-check semantics. Clear unique selected tensor
  objects once in reverse binding order under `torch.no_grad()`.

- [x] **Step 6: Run focused GREEN**

  Run the focused script and `python3 -m py_compile` for both new files.

---

### Task 2: Authorized Adapter Success and Injected-Failure Worker TDD

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_private_candidate_ownership_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_private_candidate_ownership_preflight.py`

**Interfaces:**
- Consumes:
  approved metadata, fresh CPU pool/target, exact bounded request, existing
  `build_qwen35_authorized_checkpoint_candidate_loader(...)`, and one complete
  oracle row.
- Produces:
  `run_private_candidate_ownership_rank_worker(...)` success/failure rows.

- [x] **Step 1: Write success-row RED tests**

  Require exact request fields, one provider/adapter call, consumed target,
  exact `Qwen35LoadedCheckpointCandidate`, complete hash verification, exact
  stats:

  ```python
  {
      (1, 0): (320, 320, 1, 3763655360, 1017118720),
      (2, 0): (320, 320, 1, 3763655360, 1017118720),
      (2, 1): (320, 320, 1, 3763655360, 1017118720),
  }
  ```

  Also require 320 binding hashes, 26 phase hashes, aggregate verification,
  24 aliases, cleanup, no publication, no forward, and CUDA false.

- [x] **Step 2: Write injected-failure RED tests**

  Patch only the streamed module's internal
  `_assign_qwen35_checkpoint_source_bindings` within the worker:

  ```python
  def injected(bindings, source, **kwargs):
      if assignment_calls == 0:
          result = original(bindings, source, **kwargs)
          record_first_source_hashes(bindings)
          assignment_calls += 1
          return result
      raise RuntimeError(
          "injected ownership-transfer assignment failure"
      )
  ```

  Require exact first-source hashes, no candidate return, `_consumed == true`,
  repeated `take()` rejection, cleanup, rotary/pool preservation, and restored
  production assignment function.

- [x] **Step 3: Implement real target/request/adapter construction**

  Reuse approved metadata and target construction from the loader-core gate.
  Construct:

  ```python
  Qwen35CheckpointCandidateLoadRequest(
      checkpoint_dir=APPROVED_MODEL_DIR,
      model_fingerprint=APPROVED_MODEL_MANIFEST_SHA256,
      max_tensor_bytes=1017118720,
      authorization_sha256=AUTHORIZATION_SHA256,
  )
  ```

  The provider must return the single exact target and fail if invoked twice.

- [x] **Step 4: Implement success and failure transactions**

  Snapshot target/pool/rotary state, initialize selected destinations, invoke
  the adapter once, validate result or expected injected error, and always
  restore wrappers plus clear selected destinations in `finally`.

- [x] **Step 5: Implement row validation and memory diagnostics**

  Add exact row schemas for `mode="success"` and
  `mode="injected_failure"`. Memory ceiling errors must include all observed
  and allowed total/post-Torch/post-metadata values.

- [x] **Step 6: Run focused GREEN and compile**

  Run all focused tests directly with Python 3.12 and compile with local
  Python 3.9 to preserve orchestration compatibility.

---

### Task 3: Source-Bound Six-Process Orchestration and Safety Audits

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_private_candidate_ownership_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_private_candidate_ownership_preflight.py`

**Interfaces:**
- Consumes:
  two local prerequisite artifacts, the frozen loader-core 44-file source
  closure, and six `(tp_size, tp_rank, mode)` contexts.
- Produces:
  CLI modes `run`, `internal-rank-worker`, `internal-finalize`, and
  `validate`, plus atomic local/remote artifacts.

- [x] **Step 1: Write orchestration RED tests**

  Require:

  ```text
  source files: 45 unique
  prerequisite transfers: 2 exact SHA256-bound files
  workers: 6 fresh PIDs
  modes per TP row: success + injected_failure
  CUDA_VISIBLE_DEVICES: empty
  OMP_NUM_THREADS/MKL_NUM_THREADS: 8
  finalizer: separate process
  partial worker failure: no authoritative artifacts
  ```

- [x] **Step 2: Implement deterministic source/prerequisite staging**

  Extend the loader-core source tar with only the new preflight tool. Verify
  all remote hashes and both prerequisite hashes before launching workers.

- [x] **Step 3: Implement remote workers and finalizer**

  Launch six workers in fixed order:

  ```text
  (1,0,success), (1,0,injected_failure),
  (2,0,success), (2,0,injected_failure),
  (2,1,success), (2,1,injected_failure)
  ```

  Finalize only after all rows validate. Atomically publish
  `private_candidate_ownership_preflight.json` and `source_manifest.json`
  locally and remotely.

- [x] **Step 4: Add CLI and strict path checks**

  Reject checkpoint or prerequisite paths that do not resolve to the approved
  identities. Keep JSON output deterministic and Python 3.9 compatible.

- [x] **Step 5: Run AST and source audits**

  Prove:

  ```text
  authorized adapter builder call sites: exactly 1
  direct streamed-loader call sites: 0
  direct target.take call sites: 0
  publication/ModelRunner/Engine/scheduler/step calls: 0
  forward/inference calls: 0
  CUDA calls other than is_initialized: 0
  production modules modified by this plan: 0
  ```

- [x] **Step 6: Verify closure and repository hygiene**

  Verify exact 45-file closure, exact worker hard rejection, `git diff
  --check`, and zero staged files.

---

### Task 4: Live Gate, Independent Verification, Regression, and Handoff

**Files:**
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/private_candidate_ownership_preflight.json`
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/specs/2026-07-28-qwen35-private-candidate-ownership-transfer-gate-design.md`
- Modify:
  `docs/superpowers/plans/2026-07-28-qwen35-private-candidate-ownership-transfer-gate.md`

**Interfaces:**
- Consumes: completed local gate, approved remote checkpoint, and two
  immutable prerequisites.
- Produces: authoritative proof of private ownership transfer and post-transfer
  failure cleanup, plus the next safe publication-slot boundary.

- [x] **Step 1: Run one unique source-bound six-process remote gate**

  Preserve every failed run tag. If a memory ceiling fails after cleanup,
  capture exact observed values, add a focused ceiling-contract RED test,
  calibrate to a 256-MiB boundary with at least 256 MiB headroom, and rerun
  under a new tag.

- [x] **Step 2: Independently verify emitted evidence**

  Use a standard-library-only verifier that imports neither the gate nor
  TinyLLMForge modules. Compare success hashes with the complete oracle and
  verify failure first-source hashes, stats, six unique PIDs, ownership
  transitions, cleanup, memory, and source binding.

- [x] **Step 3: Verify exact inventory and hashes**

  Require:

  ```text
  remote source files: 45
  remote root input artifacts: 2
  remote root result artifacts: 2
  local result artifacts: 2
  local/remote result SHA256: exact equality
  ```

- [x] **Step 4: Run regression matrix**

  Run focused ownership, tiled loader-core, complete transaction, target
  factory, candidate adapter, streamed loader, tiled loader, metadata, reader,
  assignment, worker request, loader construction, authorization, safety,
  ModelRunner authorized loader, and publication/binding regression scripts.

- [x] **Step 5: Record exact evidence and claim boundary**

  Append run tag, artifact hashes, source-tree hash, PIDs, memory rows, success
  stats, injected-failure source/hash evidence, independent check count,
  regression counts, static audit, failed-run inventory, what is proven, what
  remains absent, and the next private publication-slot transaction gate.

- [x] **Step 6: Final verification**

  Re-run focused tests, compile, both local and remote CLI validation, artifact
  SHA checks, AST audits, `git diff --check`, and staged-files-zero before
  marking every checkbox complete.
