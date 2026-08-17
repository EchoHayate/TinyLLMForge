# Qwen3.5 Heterogeneous Two-Layer Tile Transaction Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream, independently verify, copy, isolate, completely cover, and roll back real checkpoint tiles for linear-attention layer 0 and full-attention layer 3 at TP=1 and TP=2.

**Architecture:** Extend the completed layer-0 worker with a frozen two-layer selector and cross-layer state machine. Reuse the generic exact-range derivation and production tile-copy primitive, add per-layer aggregates and two alias groups, and publish only after three fresh rank processes pass independent source-bound validation.

**Tech Stack:** Python standard library, `os.pread`, SHA256, PyTorch CPU BF16/F32 tensors, existing Qwen3.5 binding/tile planners and copy primitive, source-bound SSH orchestration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Use only `sitian@10.232.195.203`.
- Select exactly bindings `1..14` and `227..237`.
- Require layer types `0=linear_attention`, `3=full_attention`.
- Use `max_tile_bytes=65536`.
- Open the shard exactly twice per rank.
- Retain at most one production tile, one verifier tile, and one decoded tile.
- Never call a checkpoint loader, assignment, `target.take()`, candidate
  installation, forward, CUDA, Engine, publication, or restore.
- Preserve failed/superseded evidence and schema-v2 canonical `NO_GO`.
- Do not stage, commit, merge, overwrite, or delete evidence.

---

### Task 1: Frozen Two-Layer Contract TDD

**Files:**
- Create: `tools/test_qwen35_real_checkpoint_heterogeneous_two_layer_preflight.py`
- Create: `tools/qwen35_real_checkpoint_heterogeneous_two_layer_preflight.py`

**Interfaces:**
- Consumes: completed layer-0 preflight helpers, the real 320-binding plan,
  and `build_qwen35_checkpoint_tile_plan(..., max_tile_bytes=65536)`.
- Produces: `SELECTED_BINDING_INDICES`, `UNIQUE_BINDING_ORDER`,
  `TWO_LAYER_CONTRACTS`, `binding_contract(index, tp_size)`, and exact
  validators used by later tasks.

- [x] Write a failing focused test that imports the new module and requires:
  selected bindings `1..14,227..237`; 23 unique destinations; aliases
  `[[12,13],[229,230]]`; layer order `[0,3]`; the frozen tile/kind/byte/range
  contracts for all three TP rows; and the written memory ceilings.
- [x] Run
  `/opt/homebrew/bin/python3.12 tools/test_qwen35_real_checkpoint_heterogeneous_two_layer_preflight.py`
  and confirm RED because the module is absent.
- [x] Implement only the constants, binding descriptions, and validation
  helpers needed by the first test.
- [x] Add failing tests that reject schedule drift, selected-index drift,
  alias drift, invalid gate/up slices, duplicate PIDs, counter drift, memory
  drift, and incomplete per-layer aggregates.
- [x] Implement minimal row and aggregate validators and run focused GREEN.

### Task 2: Cross-Layer Streaming Transaction TDD

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_heterogeneous_two_layer_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_heterogeneous_two_layer_preflight.py`

**Interfaces:**
- Consumes: `derive_tile_ranges`, `_tensor_bytes`, and the production
  `_copy_qwen35_checkpoint_tile`.
- Produces:
  `apply_verify_and_rollback_two_layer_tiles(...)` for synthetic tests and
  `_stream_two_layer_transaction(...)` for the real worker.

- [x] Write a synthetic failing test with two layer-0 destinations, two
  layer-3 destinations, both shared gate/up aliases, and one non-selected
  tensor. Require layer completion order `[0,3]`, layer-3 zero isolation
  before its first tile, non-selected isolation, reverse unique-object
  rollback, and all tensors zero after rollback.
- [x] Add fail-closed tests for returning to layer 0 after layer 3 starts,
  early layer-3 mutation, missing/duplicate tiles, non-shared aliasing,
  alias overlap/gap, incomplete layer bytes, short read, hash mismatch,
  non-selected mutation, and rollback failure.
- [x] Implement deterministic two-layer tile selection and the synthetic
  transaction helper.
- [x] Implement two-descriptor real streaming with per-binding, per-layer,
  and transaction rolling hashes, exact coverage counters, isolation
  checkpoints, and reverse rollback.
- [x] Run focused GREEN and `py_compile`.

### Task 3: Fresh Workers and Atomic Evidence TDD

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_heterogeneous_two_layer_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_heterogeneous_two_layer_preflight.py`

**Interfaces:**
- Produces CLI modes `run`, `internal-rank-worker`, `internal-finalize`, and
  `validate`; a 41-file deterministic tar closure; and atomic local/remote
  two-artifact publication.

- [x] Write failing tests for exact 41-file staging, three fresh rank
  processes, empty `CUDA_VISIBLE_DEVICES`, fixed thread counts, separate
  finalizer, source binding, partial-failure non-publication, and exact
  artifact names.
- [x] Implement CPU target construction without `target.take()`, exact
  layer0/layer3 schedule checks, selected tile execution, memory recording,
  row validation, aggregate validation, staging, finalization, round trip,
  and atomic publication.
- [x] Run focused GREEN and compile.
- [x] Run exact source-closure, forbidden-call AST, exact two-descriptor
  streaming AST, real-worker hard-rejection, `git diff --check`, and
  staged-file audits.

### Task 4: Live Gate, Independent Verification, and Handoff

**Files:**
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/heterogeneous_two_layer_preflight.json`
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/specs/2026-07-28-qwen35-heterogeneous-two-layer-tile-transaction-gate-design.md`
- Modify:
  `docs/superpowers/plans/2026-07-28-qwen35-heterogeneous-two-layer-tile-transaction-gate.md`

- [x] Run one unique source-bound remote gate with TP1 rank0 and TP2
  rank0/rank1 in three fresh processes.
- [x] Run an independent direct-`pread` verifier that does not import the gate
  and reproduces all 75 binding hashes, three transaction hashes, six
  per-layer hashes, TP reconstructions, replicated equality, axis-1
  interleave, segmented Q/K/V layout, and both shared destinations.
- [x] Verify exact remote inventory, source/artifact hashes, unique PIDs,
  memory ceilings, counters, isolation checkpoints, rollback, and no partial
  evidence.
- [x] Run heterogeneous-two-layer, layer0, five-transform, one-tile, CPU,
  meta, loader, metadata, reader, worker, factory, binding, authorization,
  and safety regressions.
- [x] Append exact evidence and the next safe boundary to the handoff; mark
  checkboxes only after fresh verification.

## Completed Run

```text
run tag:
  qwen35-heterogeneous-two-layer-20260728-060325
status:
  PASS
source tree SHA256:
  66ef9e8e5c12eb8b06ed419c356035773afef04cb8c7c3985320841d3f4a940e
record SHA256:
  e0cf1f0d1e48347b771aae045e22fa0b81b9dd64e50fda79d01235fddb37bad9
manifest SHA256:
  352ace9a3c34f6f4f3e97fd9ec54bd968081af36c27e14785eeb0c0a4ee4037a
fresh PIDs:
  3332932, 3335035, 3336758
independent direct-pread checks:
  115 passed
```

Fresh regression matrix:

```text
heterogeneous two-layer preflight: passed (7 tests)
layer0 transaction preflight: passed (6 tests)
five-transform bundle preflight: passed (6 tests)
one-tile payload preflight: passed (6 tests)
CPU materialization preflight: passed (6 tests)
meta target preparation preflight: passed (7 tests)
loader construction preflight: passed (5 tests)
metadata-header preflight: passed (6 tests)
bounded metadata reader: passed (4 tests)
worker helper: passed (6 tests)
candidate factory: passed (6 tests)
real component binding: passed (1 test)
loader configuration: passed (4 tests)
candidate loader: passed (5 tests)
authorization: passed
safety gate: passed (23 tests)
focused py_compile: passed
real meta-plan contract: passed for TP1 rank0 and TP2 rank0/rank1
frozen 41-file source closure: passed
forbidden-call AST scan: passed
exact two-descriptor streaming AST scan: passed
worker direct execution rejection: passed
git diff --check: passed
staged files: 0
```
