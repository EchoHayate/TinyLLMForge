# Qwen3.5 Four-Layer Cadence-Block Transaction Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream, independently verify, isolate, completely cover, and roll back all real checkpoint tiles for layers 0–3 at TP=1 and TP=2.

**Architecture:** Generalize the completed heterogeneous two-layer state machine to four ordered layers while retaining exact production binding/tile order. Track per-binding, per-layer, and transaction hashes; enforce transition isolation at layers 1, 2, and 3; and atomically publish evidence only after three fresh rank processes pass.

**Tech Stack:** Python standard library, `os.pread`, SHA256, PyTorch CPU BF16/F32 tensors, existing Qwen3.5 binding/tile planners and copy primitive, source-bound SSH orchestration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Use only `sitian@10.232.195.203`.
- Select exactly bindings `1..28`, `160..173`, and `227..237`.
- Require layer types `linear,linear,linear,full`.
- Use `max_tile_bytes=65536`.
- Open the shard exactly twice per rank.
- Retain at most one production tile, one verifier tile, and one decoded tile.
- Never call a checkpoint loader, assignment, `target.take()`, candidate
  installation, forward, CUDA, Engine, publication, or restore.
- Preserve failed/superseded evidence and schema-v2 canonical `NO_GO`.
- Do not stage, commit, merge, overwrite, or delete evidence.

---

### Task 1: Four-Layer Contract and Validation TDD

**Files:**
- Create: `tools/test_qwen35_real_checkpoint_four_layer_cadence_preflight.py`
- Create: `tools/qwen35_real_checkpoint_four_layer_cadence_preflight.py`

- [x] Write failing tests for the exact non-contiguous 53 binding indices,
  49 unique destinations, four alias groups, layer order `[0,1,2,3]`,
  aggregate/per-layer tile-range-byte contracts, and memory ceilings.
- [x] Run RED and confirm the module is absent.
- [x] Implement frozen constants and binding-to-layer mapping by exact target
  prefix, not by assumed binding arithmetic.
- [x] Add validator tests that reject schedule, index, alias, layer result,
  transition checkpoint, counter, PID, source, and memory drift.
- [x] Implement minimal row/aggregate validators and run focused GREEN.

### Task 2: Four-Layer Streaming State Machine TDD

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_four_layer_cadence_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_four_layer_cadence_preflight.py`

- [x] Write a synthetic four-layer transaction test with four gate/up alias
  pairs and one non-selected tensor.
- [x] Require exact transition checkpoints:
  completed layers changed, future layers zero, and non-selected tensors zero.
- [x] Add fail-closed tests for returning to an earlier layer, skipping a
  layer, early future-layer mutation, incomplete bytes, alias overlap/gap,
  short read, hash mismatch, non-selected mutation, and rollback failure.
- [x] Implement synthetic and real four-layer streaming transactions with
  per-binding/per-layer/aggregate hashes and reverse unique-object rollback.
- [x] Run focused GREEN and `py_compile`.

### Task 3: Fresh Workers and Atomic Evidence TDD

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_four_layer_cadence_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_four_layer_cadence_preflight.py`

- [x] Write failing tests for exact 42-file staging, three fresh rank
  processes, empty CUDA, fixed threads, separate finalizer, source binding,
  partial-failure non-publication, and exact artifact names.
- [x] Implement CPU target construction without `target.take()`, exact
  schedule/selection checks, memory recording, CLI modes, staging,
  finalization, round trip, and atomic publication.
- [x] Run focused GREEN, compile, exact closure, forbidden-call AST,
  two-descriptor streaming AST, worker hard rejection, `git diff --check`,
  and staged-file audits.

### Task 4: Live Gate, Independent Verification, and Handoff

**Files:**
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/four_layer_cadence_preflight.json`
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/specs/2026-07-28-qwen35-four-layer-cadence-block-transaction-gate-design.md`
- Modify:
  `docs/superpowers/plans/2026-07-28-qwen35-four-layer-cadence-block-transaction-gate.md`

- [x] Run one unique source-bound remote gate with three fresh rank processes.
- [x] Independently direct-`pread` and reproduce 159 binding hashes, 12
  per-layer hashes, 3 transaction hashes, all TP reconstructions, and all
  12 shared-destination partitions without importing the gate.
- [x] Verify exact inventory, source/artifact hashes, unique PIDs, memory,
  transitions, counters, isolation, rollback, and no partial evidence.
- [x] Run four-layer, two-layer, layer0, bundle, one-tile, CPU, meta, loader,
  metadata, reader, worker, factory, binding, authorization, and safety
  regressions.
- [x] Append exact evidence and the next safe boundary; mark checkboxes only
  after fresh verification.
