# Qwen3.5 Private Candidate Publication-Slot Transaction Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish one complete approved real Qwen3.5 checkpoint candidate into the existing private one-shot production publication slot at TP=1 and TP=2, then prove deterministic tensor cleanup and whole-scope object-graph discard after success and injected post-publication failure.

**Architecture:** Reuse the completed ownership gate as the exact candidate-acquisition prerequisite and the complete-checkpoint artifact as the immutable value oracle. Each fresh CPU worker loads one private candidate through the existing authorized adapter, publishes it once into a fresh `Qwen35HybridModelOwnerPublicationSlot`, validates exact visibility and hashes, clears selected tensors in `finally`, drops the entire nested scope, and proves all private publication objects are garbage-collected.

**Tech Stack:** Python standard library, PyTorch CPU, safetensors streamed loader, existing Qwen3.5 prepared-target/authorized-adapter/publication-slot modules, source-bound SSH orchestration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Use only `sitian@10.232.195.203`.
- Use complete prerequisite run `qwen35-complete-checkpoint-20260728-065128`.
- Use ownership prerequisite run `qwen35-private-ownership-20260728-090000`.
- Use exactly bindings `0..319`.
- Use request `max_tensor_bytes=1017118720`.
- Use authorization SHA256 `10a39d6eb918cb5e8d1ccf52a723cdca4db4dffb9fd4ded62b1b766474d4fde4`.
- Call the existing authorized adapter exactly once per worker.
- Construct the existing production publication slot exactly once per worker.
- Call `slot.publish(candidate)` exactly once per worker.
- Never call `target.take()` or `load_qwen35_fresh_checkpoint_candidate()` directly from the preflight.
- Never add or call slot `clear()` or `replace()`.
- Never install or bind a candidate into ModelRunner, Engine, scheduler, `LLMEngine.step()`, CUDA, forward, or inference.
- Clear and verify every selected destination after success or failure.
- Discard the whole private slot/candidate/owner/model/pool/target graph and prove weak-reference collection.
- Preserve all failed/superseded evidence and schema-v2 canonical `NO_GO`.
- Do not modify production publication, adapter, factory, streamed-loader, worker, ModelRunner, or Engine files.
- Do not stage, commit, merge, overwrite, or delete evidence.
- Do not claim production latency, throughput, cache, GPU-memory, compression, or quality benefit.

---

### Task 1: Publication Scope and Cleanup TDD

**Files:**
- Create: `tools/test_qwen35_real_checkpoint_private_publication_slot_preflight.py`
- Create: `tools/qwen35_real_checkpoint_private_publication_slot_preflight.py`

**Interfaces:**
- Consumes:
  `complete_checkpoint_transaction_preflight.json`,
  `private_candidate_ownership_preflight.json`, and one exact loaded candidate
  fixture.
- Produces:
  `load_private_publication_prerequisites(...)`,
  `validate_published_private_candidate(...)`, and
  `execute_private_publication_scope(...)`.

- [x] **Step 1: Write exact prerequisite RED tests**

  Require:

  ```python
  complete_sha256 = (
      "7ed84bd1e4e894e9ea990ff2dc631db849f805f4468a9b266bd308d690acb176"
  )
  ownership_sha256 = (
      "977a20a1986ade81e2b94063287cd15e6ece2adc3c818f3e0d9589f75b1adac4"
  )
  ownership_source_tree_sha256 = (
      "91f9225a6ee214049002dc12bc7a669cdfa6a0d847b03e0cc107834f96f561a0"
  )
  rows = ((1, 0), (2, 0), (2, 1))
  ```

  Validate exact schemas, fixed six-row ownership ordering, six unique
  prerequisite PIDs, complete success hash evidence, injected-failure hash
  evidence, cleanup, and CUDA false.

- [x] **Step 2: Run prerequisite RED**

  Run:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.12 \
    tools/test_qwen35_real_checkpoint_private_publication_slot_preflight.py
  ```

  Expected: module import failure because the publication preflight does not
  exist.

- [x] **Step 3: Implement strict prerequisite loading**

  Implement exact SHA256, schema, source-tree, row-order, PID, hash-evidence,
  ownership-transition, cleanup, memory, and CUDA validation. Select the
  success ownership row and complete oracle row for each TP tuple.

- [x] **Step 4: Write publication-scope RED tests**

  Build one compact exact candidate/target fixture and require:

  ```text
  slot before publish: candidate/owner/fingerprint all None
  publication calls: 1
  returned owner: exact candidate.owner
  slot candidate/owner/fingerprint: exact identity/value
  selected values: exact before cleanup
  selected destinations: zero after scope
  non-selected values: unchanged
  pool state: unchanged
  escaped strong objects: none
  weakrefs after gc: slot/candidate/owner/model/pool/target all dead
  ```

- [x] **Step 5: Implement nested publication scope**

  Keep all strong private references inside a nested function. Return only
  scalar/hash evidence plus weak references. Clear selected unique tensor
  objects once in reverse order under `torch.no_grad()`, exit the nested
  function, run `gc.collect()`, and require all tracked weak references to be
  dead.

- [x] **Step 6: Run focused GREEN and Python 3.9 compile**

  Run the focused script with Python 3.12 and compile both new files with
  `/usr/bin/python3 -m py_compile`.

---

### Task 2: Real Authorized Load, Publication, and Injected Failure

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_private_publication_slot_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_private_publication_slot_preflight.py`

**Interfaces:**
- Consumes:
  approved metadata, fresh CPU target, exact bounded request, existing
  authorized adapter, existing production publication slot, and one complete
  oracle row.
- Produces:
  `run_private_publication_slot_rank_worker(...)` success/failure rows.

- [x] **Step 1: Write success-row RED tests**

  Require exact request identity, one provider call, one adapter call, one
  publication call, target consumption, exact slot visibility, and:

  ```python
  loader_stats = {
      "assigned_bindings": 320,
      "source_tensors": 320,
      "shard_count": 1,
      "loaded_bytes": 3763655360,
      "peak_source_bytes": 1017118720,
  }
  ```

  Require serialized 320 binding hashes, 26 phase hashes, aggregate hash, 24
  aliases, cleanup, scope collection, no runtime installation, no forward,
  and CUDA false.

- [x] **Step 2: Write injected-post-publication RED tests**

  After exact `slot.publish(candidate)` visibility is established, raise:

  ```python
  RuntimeError("injected private publication-slot failure")
  ```

  Require one successful publication before injection, exact pre-injection
  slot candidate/owner/fingerprint evidence, no escaped object, cleanup,
  scope collection, and exact injected error.

- [x] **Step 3: Implement real target/request/adapter/slot construction**

  Reuse approved metadata and fresh CPU target construction from the ownership
  gate. Build the exact bounded request, existing authorized adapter, and one
  exact `Qwen35HybridModelOwnerPublicationSlot` inside the nested scope.

- [x] **Step 4: Implement publication transactions**

  Validate the loaded candidate before publication, publish once, validate
  exact slot visibility and graph coherence, rehash all values after
  publication, optionally inject the exact failure, and always clear selected
  destinations in `finally`.

- [x] **Step 5: Implement strict row validation and memory diagnostics**

  Define exact success and `injected_post_publication_failure` schemas. Require
  all weak-reference collection flags, exact memory recomputation, and TP
  ceilings with observed/allowed values in failures.

- [x] **Step 6: Run focused GREEN and compile**

  Run all focused tests directly with Python 3.12 and compile with local Python
  3.9.

---

### Task 3: Source-Bound Six-Process Orchestration and Safety Audits

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_private_publication_slot_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_private_publication_slot_preflight.py`

**Interfaces:**
- Consumes:
  two immutable prerequisite artifacts, the frozen ownership 45-file source
  closure, the unchanged production publication module, and six worker
  contexts.
- Produces:
  CLI modes `run`, `internal-rank-worker`, `internal-finalize`, and
  `validate`, plus atomic local/remote artifacts.

- [x] **Step 1: Write orchestration RED tests**

  Require:

  ```text
  source files: 47 unique
  ownership closure: exact 45-file prerequisite match
  production publication module: frozen local/remote SHA
  prerequisite transfers: 2 exact SHA256-bound files
  workers: 6 fresh PIDs
  modes per TP row: success + injected_post_publication_failure
  CUDA_VISIBLE_DEVICES: empty
  OMP_NUM_THREADS/MKL_NUM_THREADS: 8
  finalizer: separate process
  partial worker failure: no authoritative artifacts
  ```

- [x] **Step 2: Implement deterministic source/prerequisite staging**

  Stage the exact 47-file tar and the two immutable prerequisites. Verify all
  local/remote source hashes and both prerequisite hashes before worker launch.

- [x] **Step 3: Implement six workers and finalizer**

  Launch in fixed order:

  ```text
  (1,0,success), (1,0,injected_post_publication_failure),
  (2,0,success), (2,0,injected_post_publication_failure),
  (2,1,success), (2,1,injected_post_publication_failure)
  ```

  Finalize only after every row validates. Atomically publish
  `private_publication_slot_preflight.json` and `source_manifest.json` locally
  and remotely.

- [x] **Step 4: Add CLI and strict path checks**

  Reject checkpoint and prerequisite paths that do not resolve to approved
  identities. Preserve deterministic JSON and Python 3.9 compatibility.

- [x] **Step 5: Run AST and source audits**

  Prove:

  ```text
  authorized adapter builder call sites: exactly 1
  production publication-slot constructor call sites: exactly 1
  slot.publish call sites: exactly 1
  direct streamed-loader call sites: 0
  direct target.take call sites: 0
  clear/replace call sites: 0
  ModelRunner/Engine/scheduler/step calls: 0
  forward/inference calls: 0
  CUDA calls other than is_initialized: 0
  production modules modified by this plan: 0
  ```

- [x] **Step 6: Verify closure and repository hygiene**

  Verify exact 47-file closure, exact ownership closure inheritance, unchanged
  publication-module SHA, exact worker hard rejection, `git diff --check`, and
  zero staged files.

---

### Task 4: Live Gate, Independent Verification, Regression, and Handoff

**Files:**
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/private_publication_slot_preflight.json`
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/specs/2026-07-28-qwen35-private-candidate-publication-slot-transaction-gate-design.md`
- Modify:
  `docs/superpowers/plans/2026-07-28-qwen35-private-candidate-publication-slot-transaction-gate.md`

**Interfaces:**
- Consumes:
  completed local publication gate, approved remote checkpoint, and two
  immutable prerequisites.
- Produces:
  authoritative proof of private publication visibility and whole-scope
  discard, plus the next dependency-light ModelRunner-local boundary.

- [x] **Step 1: Run one unique source-bound remote gate**

  Preserve every failed run tag. If a memory ceiling fails after cleanup,
  capture exact values, add a focused ceiling-contract RED test, calibrate to a
  256-MiB boundary with at least 256 MiB headroom, and rerun under a new tag.

- [x] **Step 2: Independently verify emitted evidence**

  Use a standard-library-only verifier importing neither the gate nor
  TinyLLMForge. Compare all success hashes with the complete oracle and verify
  publication identity, exact injected error, six unique PIDs, cleanup,
  weak-reference collection, memory, and source binding.

- [x] **Step 3: Verify exact inventory and hashes**

  Require:

  ```text
  remote source files: 47
  remote root input artifacts: 2
  remote root result artifacts: 2
  local result artifacts: 2
  local/remote result SHA256: exact equality
  ```

- [x] **Step 4: Run regression matrix**

  Run focused publication, ownership transfer, hybrid model publication,
  tiled loader-core, complete transaction, target factory, candidate adapter,
  streamed loader, tiled loader, metadata, reader, assignment, worker request,
  loader construction, authorization, safety, ModelRunner published-candidate
  binding, and Engine all-rank binding scripts.

- [x] **Step 5: Record exact evidence and claim boundary**

  Append run tag, artifact/source hashes, PIDs, memory rows, publication
  identity evidence, success hashes, injected-failure evidence, independent
  check count, regression counts, static audit, failed-run inventory, proven
  boundary, absent runtime paths, and the next safe gate.

- [x] **Step 6: Final verification**

  Re-run focused tests, Python 3.9 compile, local/remote CLI validation,
  independent verifier, artifact SHA checks, AST/source audits,
  `git diff --check`, and staged-files-zero before marking every checkbox
  complete.
