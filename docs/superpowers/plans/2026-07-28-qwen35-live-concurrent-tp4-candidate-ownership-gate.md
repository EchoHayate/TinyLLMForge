# Qwen3.5 Live Concurrent TP4 Candidate Ownership Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove four real Qwen3.5 TP4 rank-local checkpoint candidates remain live simultaneously and are then released without leaks under an explicit aggregate CPU-memory contract.

**Architecture:** Reuse the frozen serial producer internals but split candidate lifetime into retain and release phases. Four fresh workers are spawned together, checkpoint loads are started one rank at a time, every completed candidate remains resident behind a control-channel barrier, and an atomic coordinator snapshot proves all four PIDs and candidates are live before reverse-order release.

**Tech Stack:** Python 3.9+ standard library, PyTorch CPU, safetensors bounded streaming, `multiprocessing`, duplex pipes, `/proc` process accounting, AST extraction, SSH source-bound orchestration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Use only `sitian@10.232.195.203`.
- Preserve all failed and superseded local/remote evidence.
- Do not stage, commit, merge, create a PR, overwrite evidence, or delete evidence.
- Use provenance `real-checkpoint-derived-live-concurrent-tp4-ownership`.
- Use claim boundary `not-constructed-engine-runtime-binding`.
- Never import or construct production `LLMEngine` or `ModelRunner`.
- Never call scheduler, `LLMEngine.step()`, CUDA operations, forward, or inference.
- Stagger checkpoint loading in exact rank order `(0, 1, 2, 3)`.
- Retain every ready candidate until all four are concurrently live.
- Release in exact reverse order `(3, 2, 1, 0)`.
- Preserve the exact real worker hard rejection and schema-v2 canonical `NO_GO`.
- Do not claim accuracy, quality, latency, throughput, cache, memory, or compression benefit.

---

### Task 1: Retained Candidate Scope Contract

**Files:**
- Create: `tools/qwen35_tp4_live_concurrent_candidate_ownership_preflight.py`
- Create: `tools/test_qwen35_tp4_live_concurrent_candidate_ownership_preflight.py`

**Interfaces:**
- Consumes: pristine serial oracle, approved checkpoint identities, frozen production methods, TP4 producer component factory.
- Produces: `prepare_retained_tp4_candidate(...) -> RetainedCandidateScope` and strict ready/released row validators.

- [x] **Step 1: Write retained-scope RED tests**

  Require rank-specific real payload validation, one load/publish/bind call,
  all retained object references live before release, selected tensors
  non-zero before release, and CUDA/forward counters unchanged.

- [x] **Step 2: Run RED**

  Run:

  ```bash
  python3 tools/test_qwen35_tp4_live_concurrent_candidate_ownership_preflight.py
  ```

  Require failure because the preflight module is absent.

- [x] **Step 3: Implement retained scope**

  Factor the existing producer lifetime into:

  ```python
  retained = prepare_retained_tp4_candidate(...)
  ready_row = retained.ready_row()
  released_row = retained.release()
  ```

  `release()` is single-use and performs reverse unique-object clear,
  invariant validation, reference dropping, garbage collection, and exact
  collection reporting.

- [x] **Step 4: Add release failure tests**

  Cover duplicate release, clear failure, escaped owner, changed non-selected
  tensor, pool mutation, and invalid ready-row payload identity.

- [x] **Step 5: Run focused GREEN and compile**

  Require all retained-scope tests and Python 3.9 compilation to pass.

### Task 2: Four-Worker Staggered Residency Coordinator

**Files:**
- Modify: `tools/qwen35_tp4_live_concurrent_candidate_ownership_preflight.py`
- Modify: `tools/test_qwen35_tp4_live_concurrent_candidate_ownership_preflight.py`

**Interfaces:**
- Consumes: retained scope and four duplex control channels.
- Produces: canonical ready rows, concurrent snapshot, released rows, and strict ordering evidence.

- [x] **Step 1: Write coordinator RED tests**

  Require all workers spawned before loading, starts `(0,1,2,3)`, one ready
  acknowledgement before the next start, four simultaneously live PIDs,
  zero premature releases, snapshot before release, releases `(3,2,1,0)`,
  and all workers joined.

- [x] **Step 2: Implement worker protocol**

  Freeze messages:

  ```text
  START -> READY -> RELEASE -> RELEASED
  ABORT -> RELEASED
  ```

  Reject duplicate, missing, out-of-order, wrong-rank, or non-canonical
  messages.

- [x] **Step 3: Implement `/proc` residency snapshot**

  Read each worker's `State`, `VmRSS`, and `VmHWM` only after all four ready
  rows validate. Require all four PIDs alive and no release acknowledgement.

- [x] **Step 4: Add directed failure RED tests**

  Cover early exit, participant mismatch, premature release, stalled release,
  load overlap, and PID reuse.

- [x] **Step 5: Implement fail-closed abort and GREEN**

  On any failure, issue `ABORT` to every live worker, wait for bounded graceful
  cleanup, then terminate only workers that exceed the cleanup timeout.

### Task 3: Aggregate Memory and Artifact Contract

**Files:**
- Modify: `tools/qwen35_tp4_live_concurrent_candidate_ownership_preflight.py`
- Modify: `tools/test_qwen35_tp4_live_concurrent_candidate_ownership_preflight.py`
- Create: `tools/verify_qwen35_tp4_live_concurrent_candidate_ownership_gate.py`
- Create: `tools/test_verify_qwen35_tp4_live_concurrent_candidate_ownership_gate.py`

**Interfaces:**
- Consumes: four ready/released rows and concurrent snapshot.
- Produces: immutable ownership artifact, source manifest, and independent verification.

- [x] **Step 1: Write memory-contract RED tests**

  Require:

  ```text
  per-worker total VmHWM increment <= 3145728 KiB
  aggregate worker VmHWM increment <= 12582912 KiB
  aggregate ready VmRSS <= 8388608 KiB
  host MemAvailable decrease <= 12582912 KiB
  preflight MemAvailable >= 16777216 KiB
  ```

- [x] **Step 2: Implement memory validation**

  Recompute every delta from raw observations. Include actual and allowed
  values in every failure. Reject negative deltas and inconsistent worker
  versus coordinator `/proc` observations.

- [x] **Step 3: Implement atomic finalization**

  Publish only after four valid released rows and all PIDs are absent. Preserve
  ready rows and the concurrent snapshot byte-for-byte.

- [x] **Step 4: Implement stdlib-only verifier**

  Independently validate source/prerequisite/method hashes, payload identities,
  live overlap, ordering, memory, release, cleanup, and inventory without
  importing TinyLLMForge or the gate.

- [x] **Step 5: Add tamper tests**

  Reject missing live PID, re-signed production import, payload drift,
  premature release, start/release reorder, memory re-signing, PID overlap,
  and incomplete collection.

### Task 4: Source-Bound Remote Gate and Audit

**Files:**
- Modify: `tools/qwen35_tp4_live_concurrent_candidate_ownership_preflight.py`
- Modify: `tools/test_qwen35_tp4_live_concurrent_candidate_ownership_preflight.py`
- Modify: `tools/verify_qwen35_tp4_live_concurrent_candidate_ownership_gate.py`
- Modify: `tools/test_verify_qwen35_tp4_live_concurrent_candidate_ownership_gate.py`
- Create: `experiments/qwen35_hybrid_state/<run-tag>/tp4_live_concurrent_candidate_ownership.json`
- Create: `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `docs/superpowers/specs/2026-07-28-qwen35-live-concurrent-tp4-candidate-ownership-gate-design.md`
- Modify: `docs/superpowers/plans/2026-07-28-qwen35-live-concurrent-tp4-candidate-ownership-gate.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: exact source closure, pristine serial prerequisite, approved checkpoint, and live host memory preflight.
- Produces: one authoritative remote run plus independent local/remote verification.

- [x] **Step 1: Implement source-bound CLI**

  Add `run`, `internal-worker`, `internal-finalize`, and `validate` modes.
  Stage deterministic sources and prerequisites, rehash remotely, and refuse
  execution when host memory preflight fails.

- [x] **Step 2: Run focused and adjacent regressions**

  Run retained-scope/coordinator/verifier tests plus serial provenance,
  checkpoint assignment/binding, candidate factory, full-attention shell,
  loader configuration, worker rejection, and schema-v2 `NO_GO`.

- [x] **Step 3: Run one authoritative remote tag**

  Use the exact remote Python and a unique tag. Preserve every failed tag.
  Do not place the verifier inside the authoritative run's `source/`.

- [x] **Step 4: Verify local and remote evidence**

  Require exact result SHA equality, source inventory, four ready and four
  released rows, one concurrent snapshot, zero residual PIDs, all selected
  tensors cleared, and all private objects collected.

- [x] **Step 5: Complete requirement audit**

  Map every spec requirement to concrete artifact fields and test/verifier
  evidence. Record what the gate proves and does not prove. Keep the long-term
  performance goal active and set the next TODO to constructed
  Engine/ModelRunner ownership without scheduler or forward.
