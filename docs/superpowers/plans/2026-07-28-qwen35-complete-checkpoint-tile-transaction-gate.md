# Qwen3.5 Complete Checkpoint Tile Transaction Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream, independently verify, isolate, completely cover, and roll back all 320 real Qwen3.5 checkpoint bindings at TP=1 and TP=2.

**Architecture:** Generalize the completed four-layer transaction to a frozen 26-phase ledger that follows real binding-plan order rather than numerical layer order. Track per-binding, per-phase, per-layer, root, and transaction hashes; snapshot all 296 unique destinations; and atomically publish evidence only after three fresh rank processes pass.

**Tech Stack:** Python standard library, `os.pread`, SHA256, PyTorch CPU BF16/F32 tensors, existing Qwen3.5 binding/tile planners and copy primitive, source-bound SSH orchestration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Use only `sitian@10.232.195.203`.
- Select exactly bindings `0..319`.
- Preserve the exact frozen 26-phase binding-plan order.
- Use `max_tile_bytes=65536`.
- Open the shard exactly twice per rank.
- Retain at most one production tile, one verifier tile, and one decoded tile.
- Never call a checkpoint loader, assignment, `target.take()`, candidate
  installation, forward, CUDA, Engine, publication, or restore.
- Preserve failed/superseded evidence and schema-v2 canonical `NO_GO`.
- Do not stage, commit, merge, overwrite, or delete evidence.

---

### Task 1: Complete Contract and Phase-Ledger TDD

**Files:**
- Create: `tools/test_qwen35_real_checkpoint_complete_transaction_preflight.py`
- Create: `tools/qwen35_real_checkpoint_complete_transaction_preflight.py`

**Interfaces:**
- Consumes: the frozen four-layer checkpoint identities and source closure.
- Produces: `PHASE_BINDING_RUNS`, `ALIAS_GROUPS`,
  `COMPLETE_TRANSACTION_CONTRACTS`, `binding_contract()`,
  `validate_complete_checkpoint_row()`, and
  `validate_complete_checkpoint_preflight()`.

- [x] Write failing tests for 320 bindings, 296 unique destinations, the exact
  26 phase runs, 24 alias groups, root binding contracts, TP aggregate
  contracts, and memory ceilings.
- [x] Run RED and confirm the new module is absent.
- [x] Implement frozen constants and target-prefix phase mapping. Never sort by
  numerical layer index.
- [x] Add validator tests that reject phase order, schedule, index, root,
  alias, binding, phase result, layer result, transition, counter, PID,
  source, hash, and memory drift.
- [x] Implement minimal validators and run focused GREEN.

### Task 2: Complete Streaming Transaction TDD

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_complete_transaction_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_complete_transaction_preflight.py`

**Interfaces:**
- Consumes: `PHASE_BINDING_RUNS`, exact binding-to-phase mapping, production
  tiles, and registered CPU tensors.
- Produces: `apply_verify_and_rollback_complete_tiles()` and
  `_stream_complete_checkpoint_transaction()`.

- [x] Write a synthetic 26-phase transaction test with root endpoints,
  representative linear/full phases, 24 alias partitions, and one
  non-selected tensor.
- [x] Require 25 exact transition checkpoints: all completed phases changed,
  all future phases zero.
- [x] Add fail-closed tests for returning to an earlier phase, numerical layer
  sorting, skipping a phase, future mutation, incomplete bytes, alias
  overlap/gap, short read, hash mismatch, non-selected mutation, root drift,
  and rollback failure.
- [x] Implement per-binding/per-phase/per-layer/root/aggregate hashes, exact
  coverage counters, two-descriptor streaming, and reverse unique-object
  rollback.
- [x] Run focused GREEN and `py_compile`.

### Task 3: Fresh Workers and Atomic Evidence TDD

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_complete_transaction_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_complete_transaction_preflight.py`

**Interfaces:**
- Consumes: approved checkpoint identity, 43-file source closure, three TP
  contexts, and the complete streaming transaction.
- Produces: CLI modes `run`, `internal-rank-worker`, `internal-finalize`, and
  `validate`; atomic local and remote evidence publication.

- [x] Write failing tests for exact 43-file staging, three fresh processes,
  empty CUDA, fixed threads, separate finalizer, exact artifact names,
  source binding, and partial-failure non-publication.
- [x] Implement CPU target construction without `target.take()`, complete
  schedule/selection checks, memory recording, staging, finalization, round
  trip, and atomic publication.
- [x] Run focused GREEN, compile, exact closure, forbidden-call AST,
  two-descriptor streaming AST, worker hard rejection, `git diff --check`,
  and staged-file audits.

### Task 4: Live Gate, Independent Verification, and Handoff

**Files:**
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/complete_checkpoint_transaction_preflight.json`
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/specs/2026-07-28-qwen35-complete-checkpoint-tile-transaction-gate-design.md`
- Modify:
  `docs/superpowers/plans/2026-07-28-qwen35-complete-checkpoint-tile-transaction-gate.md`

**Interfaces:**
- Consumes: the complete local gate and approved remote checkpoint.
- Produces: authoritative source-bound evidence and the next production-loader
  safety boundary.

- [x] Run one unique source-bound remote gate with three fresh rank processes.
- [x] Independently direct-`pread` and reproduce 960 binding hashes, 78 phase
  hashes, 3 transaction hashes, all 320 TP reconstructions, and all 72
  rank-local alias partitions without importing the gate.
- [x] Verify exact inventory, source/artifact hashes, unique PIDs, memory,
  transitions, counters, isolation, rollback, and no partial evidence.
- [x] Run complete, four-layer, two-layer, layer0, bundle, one-tile, CPU,
  meta, loader, metadata, reader, worker, factory, binding, authorization,
  and safety regressions.
- [x] Append exact evidence and the next safe boundary; mark checkboxes only
  after fresh verification.

## Completion Evidence

Authoritative source-bound run:

```text
qwen35-complete-checkpoint-20260728-065128
```

Authoritative outputs:

```text
experiments/qwen35_hybrid_state/
qwen35-complete-checkpoint-20260728-065128/
```

Fresh rank processes:

```text
TP=1 rank0 PID 3946836
TP=2 rank0 PID 3960911
TP=2 rank1 PID 3966499
```

Independent standard-library-only direct-`pread` verification passed:

```text
960 TP-local binding hashes
78 phase hashes
3 transaction hashes
320 TP reconstructions
72 rank-local alias partitions
1433 checks total
```

Authoritative hashes:

```text
source tree:
  da665b2de0aaa6533e55be0469c76ed39d92e817aabf80618f07b7efa7ef7042
complete_checkpoint_transaction_preflight.json:
  7ed84bd1e4e894e9ea990ff2dc631db849f805f4468a9b266bd308d690acb176
source_manifest.json:
  9513c3d329b4bd310158416194673d5b60faa118983c88c674c984a9b9d6bd9e
```
