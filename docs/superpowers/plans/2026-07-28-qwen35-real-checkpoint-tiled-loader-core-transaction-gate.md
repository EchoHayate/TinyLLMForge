# Qwen3.5 Real Checkpoint Tiled Loader-Core Transaction Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the production tiled loader core against all 320 approved real Qwen3.5 checkpoint bindings on fresh private CPU candidates at TP=1 and TP=2, verify exact values, and clear/discard every candidate without consuming or publishing it.

**Architecture:** Reuse the completed 320-binding transaction artifact as the immutable hash oracle. Build fresh prepared CPU targets, invoke `_load_qwen35_candidate_with_tile_plan()` exactly once per rank without `target.take()`, verify complete loaded state and loader stats, then clear all unique destinations in a fail-closed `finally` transaction.

**Tech Stack:** Python standard library, PyTorch CPU, safetensors production loader core, existing Qwen3.5 metadata/factory/tile modules, source-bound SSH orchestration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Use only `sitian@10.232.195.203`.
- Use prerequisite run `qwen35-complete-checkpoint-20260728-065128`.
- Select exactly bindings `0..319` and `max_tile_bytes=65536`.
- Invoke `_load_qwen35_candidate_with_tile_plan()` exactly once per rank.
- Never call `target.take()`, the authorized adapter, candidate installation,
  ModelRunner, Engine, publication, forward, CUDA, or inference.
- Clear and verify every private checkpoint destination after success or
  failure; preserve unbound rotary buffers exactly.
- Preserve failed/superseded evidence and schema-v2 canonical `NO_GO`.
- Do not stage, commit, merge, overwrite, or delete evidence.

---

### Task 1: Oracle and Cleanup Transaction TDD

**Files:**
- Create: `tools/test_qwen35_real_checkpoint_tiled_loader_core_preflight.py`
- Create: `tools/qwen35_real_checkpoint_tiled_loader_core_preflight.py`

**Interfaces:**
- Consumes: the authoritative complete-checkpoint artifact and a fresh prepared
  target.
- Produces: `load_complete_gate_oracle()`,
  `validate_loaded_private_candidate()`, and
  `execute_and_clear_tiled_loader_core()`.

- [x] Write a failing test that accepts only the exact prerequisite schema,
  source tree, artifact hash, three TP rows, 320 binding results, 26 phase
  results, and unique PIDs.
- [x] Run RED and confirm the module is absent.
- [x] Implement immutable oracle parsing and exact `(tp_size, tp_rank)` row
  selection.
- [x] Write synthetic fresh-target tests for success cleanup and an injected
  mid-load copy failure.
- [x] Require selected-only zero initialization before payload access, exact
  destination/rotary identity and storage preservation, `_consumed == false`,
  reverse unique-object clearing, selected all-zero state, and unbound rotary
  value preservation.
- [x] Implement the minimal execute/verify/clear wrapper and run focused GREEN.

### Task 2: Real Production Loader-Core Worker TDD

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_tiled_loader_core_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_tiled_loader_core_preflight.py`

**Interfaces:**
- Consumes: approved metadata, fresh CPU pool/target, production tile plan,
  and one oracle row.
- Produces: `run_tiled_loader_core_rank_worker()` and rank rows containing
  exact hashes, stats, memory, ownership, and cleanup evidence.

- [x] Write failing tests for one exact production-core call, no `target.take`,
  exact 320 binding/26 phase/aggregate verification, exact stats, all 24 alias
  groups, no non-selected mutation, and post-run zero state.
- [x] Implement approved metadata and target construction using the same
  architecture schedule and source identities as the complete gate.
- [x] Build the 65,536-byte production tile plan and validate exact TP
  contracts before payload access.
- [x] Invoke `_load_qwen35_candidate_with_tile_plan()` exactly once and verify
  the returned candidate retains the exact model, binding plan, tile plan, and
  model fingerprint.
- [x] Hash loaded binding views in exact binding order, derive phase and
  aggregate hashes, compare with the oracle, then clear in `finally`.
- [x] Record conservative memory deltas and prove CUDA false before and after.
- [x] Run focused GREEN and `py_compile`.

### Task 3: Source-Bound Remote Orchestration and Safety Audits

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_tiled_loader_core_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_tiled_loader_core_preflight.py`

**Interfaces:**
- Consumes: the local prerequisite artifact, 44-file source closure, and three
  TP contexts.
- Produces: CLI modes `run`, `internal-rank-worker`, `internal-finalize`, and
  `validate`, plus atomic local/remote artifacts.

- [x] Write failing tests for exact 44-file staging, prerequisite artifact
  transfer/hash binding, three fresh rank processes, empty CUDA, fixed
  threads, separate finalizer, exact artifact names, and partial-failure
  non-publication.
- [x] Implement source and prerequisite staging, remote hash verification,
  finalization, source manifest, artifact round trip, and atomic local
  publication.
- [x] Add CLI modes and reject any checkpoint or prerequisite path that does
  not match the approved identities.
- [x] Run focused GREEN and compile.
- [x] Run AST audits proving no `safe_open`, `get_slice`, direct copy loop,
  `target.take()`, adapter, ModelRunner, Engine, publication, forward, or CUDA
  allocation/operator calls, and exactly one production loader-core call
  site. Permit only `torch.cuda.is_initialized()` as a read-only observation.
- [x] Verify worker hard rejection, exact 44-file closure, `git diff --check`,
  and staged files zero.

### Task 4: Live Gate, Regression Matrix, and Handoff

**Files:**
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/tiled_loader_core_preflight.json`
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/specs/2026-07-28-qwen35-real-checkpoint-tiled-loader-core-transaction-gate-design.md`
- Modify:
  `docs/superpowers/plans/2026-07-28-qwen35-real-checkpoint-tiled-loader-core-transaction-gate.md`

**Interfaces:**
- Consumes: the completed local gate and approved remote checkpoint.
- Produces: authoritative proof of production tiled loader-core correctness
  and the next safe candidate-ownership boundary.

- [x] Run one unique source-bound remote gate with three fresh rank processes.
- [x] Independently compare every emitted binding, phase, and aggregate hash
  with the immutable complete-gate artifact and verify exact loader stats,
  unique PIDs, memory, ownership, clearing, and no partial evidence.
- [x] Verify exact remote inventory and local/remote source/artifact hashes.
- [x] Run tiled-loader-core, complete, four-layer, two-layer, layer0, bundle,
  one-tile, CPU, meta, tiled loader, loader construction, metadata, reader,
  worker, factory, binding, authorization, and safety regressions.
- [x] Append exact evidence and claim boundary; mark checkboxes only after
  fresh verification.

