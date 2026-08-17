# Qwen3.5 Five-Transform Payload Bundle Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Independently read, verify, copy, isolate, and reverse-roll back five fixed layer-0 real checkpoint tiles covering replicated, axis-0, segmented-axis-0, squeeze-axis-0, and axis-1 layouts for TP=1 and TP=2.

**Architecture:** Extend the proven one-tile fresh-process gate with a frozen five-tile contract. Use two read-only file descriptors per rank, issue exact `pread` calls for every frozen contiguous range, concatenate the TP=2 axis-1 row spans, copy only through the production tile-copy primitive, then prove simultaneous isolation and reverse rollback before atomic publication.

**Tech Stack:** Python standard library, `os.pread`, SHA256, PyTorch CPU BF16/F32 tensors, existing Qwen3.5 tile planner and copy primitive, source-bound SSH orchestration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Use only `sitian@10.232.195.203`.
- Select only binding indices `[3,4,7,9,11]`.
- Use the exact range lists and byte counts in the written design.
- Use exactly two shard file descriptors per rank.
- Never call a checkpoint loader, assignment, `target.take()`, candidate
  installation, forward, CUDA, Engine, publication, or restore.
- Preserve failed/superseded evidence and schema-v2 canonical `NO_GO`.
- Do not stage, commit, merge, overwrite, or delete evidence.

---

### Task 1: Frozen Bundle Contract and Range Reader TDD

**Files:**
- Create: `tools/test_qwen35_real_checkpoint_five_transform_bundle_preflight.py`
- Create: `tools/qwen35_real_checkpoint_five_transform_bundle_preflight.py`

**Interfaces:**
- Consumes:
  `tools/qwen35_real_checkpoint_one_tile_payload_preflight.py`,
  `build_qwen35_checkpoint_tile_plan`,
  `_copy_qwen35_checkpoint_tile`.
- Produces:
  `BUNDLE_CONTRACTS`,
  `read_and_verify_exact_ranges(path, tile_ranges)`,
  `validate_five_transform_bundle_row(row)`.

- [x] Write failing tests that require the exact 39-file source closure,
  binding indices `[3,4,7,9,11]`, exact TP/rank shapes, slices, ranges,
  per-pass bytes, and per-pass range counts.
- [x] Run the focused test and confirm failure because the bundle module is
  absent.
- [x] Implement frozen constants and range validation. Reject empty,
  unsorted, overlapping, non-positive, and aggregate-byte-mismatched ranges.
- [x] Implement exactly two descriptor opens. For each descriptor, issue one
  `os.pread` per range, fail on every short read, concatenate per-tile bytes,
  and require production/verifier bytes and SHA256 to match.
- [x] Add sparse-file fixtures covering contiguous, multi-range axis-1,
  short-read, overlap, ordering, and hash disagreement failures.
- [x] Run focused GREEN and `py_compile`.

### Task 2: Five-Tile Copy, Isolation, and Reverse Rollback TDD

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_five_transform_bundle_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_five_transform_bundle_preflight.py`

**Interfaces:**
- Consumes:
  exact production `Qwen35CheckpointTile` values and decoded source tensors.
- Produces:
  `copy_verify_and_reverse_rollback_bundle(tiles, source_tensors,
  unique_tensors)`.

- [x] Write failing synthetic tests with five distinct destination objects,
  mixed BF16/F32 dtypes, one non-selected tensor, and exact expected reverse
  rollback order.
- [x] Require fail-closed behavior for duplicate destination objects,
  initially nonzero storage, source/destination hash mismatch, mutation of a
  not-yet-selected destination, non-selected mutation, and incomplete
  rollback.
- [x] Implement copy in fixed binding order via
  `_copy_qwen35_checkpoint_tile`, simultaneous selected hash verification,
  non-selected isolation checks, reverse-order snapshot restore, and final
  all-zero verification.
- [x] Run focused GREEN.

### Task 3: Fresh-Process Worker and Atomic Orchestration TDD

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_five_transform_bundle_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_five_transform_bundle_preflight.py`

**Interfaces:**
- Produces CLI modes:
  `run`, `internal-rank-worker`, `internal-finalize`, `validate`.

- [x] Write failing tests for deterministic 39-file staging, three fresh rank
  processes, empty `CUDA_VISIBLE_DEVICES`, fixed thread counts, separate
  finalizer, exact source binding, partial-failure non-publication, and atomic
  two-artifact publication.
- [x] Implement a worker that reuses CPU materialization, derives and validates
  the five exact production tiles, performs the two-pass range reads, decodes
  exact dtypes/shapes, copies, verifies, rolls back, and records bounded memory.
- [x] Implement row and aggregate validators with exact keys, counters,
  byte/range totals, unique PIDs, source hashes, and memory ceilings.
- [x] Implement staging, finalization, remote artifact round trip, local atomic
  publication, and validation CLI.
- [x] Run focused GREEN, `py_compile`, exact 39-file closure, forbidden-call
  AST, exact two-open/range-loop AST, real-worker rejection,
  `git diff --check`, and staged-file audits.

### Task 4: Live Gate, Independent Verification, and Handoff

**Files:**
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/five_transform_bundle_preflight.json`
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/specs/2026-07-28-qwen35-five-transform-payload-bundle-gate-design.md`
- Modify:
  `docs/superpowers/plans/2026-07-28-qwen35-five-transform-payload-bundle-gate.md`

- [x] Run one unique source-bound remote gate with three independent rank
  processes.
- [x] Independently `pread` every frozen range on the remote host and verify
  all per-tile hashes, aggregate bytes, TP slicing, replicated equality,
  axis-0 concatenation, squeeze concatenation, segmented rank offsets, and
  axis-1 column interleaving.
- [x] Verify exact remote inventory, local/remote/record source hashes,
  artifact SHA256, unique PIDs, memory ceilings, zero execution counters,
  isolation, and reverse rollback.
- [x] Run bundle, one-tile, CPU materialization, meta target, loader
  construction, metadata, reader, worker, factory, binding, loader,
  authorization, and safety regressions.
- [x] Append exact evidence and the next bounded gate to
  `AGENT_HANDOFF_STATE.md`; mark checkboxes only after fresh verification.

## Completion Evidence

Authoritative run:

```text
qwen35-five-transform-bundle-20260727-210712
```

Artifact SHA256:

```text
five_transform_bundle_preflight.json:
  b6d826ff8190981cf73e151918fb40c7713eccb0dece74783ed1fbc02e5e829c
source_manifest.json:
  5f6ef341917c92ee4577b85be813fb95ae39c6bb459e3aae75b13c0a4acd0e01
```

Source tree SHA256:

```text
dd10667cc4a638d6a8d658a0f78bce72f14324ace7e8576961c4b20195226a32
```

Fresh rank evidence:

```text
TP=1 rank0:
  PID 2618752
  production/verifier/logical bytes 176672/176672/353344
  open/pread count 2/10
  total/post-Torch/post-metadata VmHWM increment:
    4213620 / 3869616 / 3738856 KiB

TP=2 rank0:
  PID 2620196
  production/verifier/logical bytes 152080/152080/304160
  open/pread count 2/28
  total/post-Torch/post-metadata VmHWM increment:
    2346352 / 2002012 / 1871764 KiB

TP=2 rank1:
  PID 2621492
  production/verifier/logical bytes 152080/152080/304160
  open/pread count 2/28
  total/post-Torch/post-metadata VmHWM increment:
    2346756 / 2002348 / 1872072 KiB
```

Independent verification:

```text
15/15 frozen tile hashes reproduced by direct remote pread
TP2 squeeze rows concatenate to TP1
TP2 axis0 values concatenate to TP1
replicated F32 bytes equal across TP1/TP2 rows
segmented TP1 first tile equals TP2 rank0 first tile
TP2 axis1 row-local column halves reconstruct TP1 rows
five destination objects remained distinct
not-yet-selected and non-selected tensors remained zero
rollback order [11,9,7,4,3]
all snapshots restored and every unique tensor zero after rollback
```

Fresh final verification:

```text
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
frozen 39-file source closure: passed
forbidden-call AST scan: passed
exact two-descriptor multi-range AST scan: passed
real worker direct-execution rejection: passed
git diff --check: passed
staged files: 0
```

Exact claim boundary:

```text
five representative real tile layout families:
  proven for TP=1/2 all ranks
non-contiguous TP axis1 exact range reconstruction:
  proven
simultaneous destination isolation and reverse rollback:
  proven
complete layer / all-binding checkpoint load:
  not executed
loader / assignment / target.take() / candidate installation:
  not executed
CUDA / Engine / publication / inference:
  absent
production speed / cache / GPU-memory / compression / quality benefit:
  unmeasured
schema-v2 canonical NO_GO:
  unchanged
```
