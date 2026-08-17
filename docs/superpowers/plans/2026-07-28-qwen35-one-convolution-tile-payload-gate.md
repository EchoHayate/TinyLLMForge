# Qwen3.5 One-Convolution-Tile Payload Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read, independently verify, copy, and roll back exactly one real layer-0 convolution payload tile for TP=1/rank0 and TP=2/rank0-1 without calling any checkpoint loader or assignment transaction.

**Architecture:** Reuse the bounded CPU materialization worker, derive one frozen `squeeze_axis0` tile from the real binding plan, perform two exact `os.pread` reads through separate file descriptors, copy with the existing tile-copy primitive, prove isolation and rollback, then publish only after three fresh processes pass.

**Tech Stack:** Python standard library, `os.pread`, SHA256, PyTorch BF16 views, existing Qwen3.5 tile planner/copy primitive, SSH source-bound orchestration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Use only `sitian@10.232.195.203`.
- Read only binding index `3`, source
  `model.language_model.layers.0.linear_attn.conv1d.weight`.
- TP=1 tile bytes `49152`; TP=2 tile bytes `24576`.
- Absolute ranges:
  TP1 `[1017209840,1017258992)`,
  TP2 rank0 `[1017209840,1017234416)`,
  TP2 rank1 `[1017234416,1017258992)`.
- Use exactly two file descriptors and two exact `pread` calls per row.
- Never call a loader, assignment, `target.take()`, candidate factory,
  forward, CUDA, Engine, publication, or restore.
- Preserve failed/superseded evidence and schema-v2 canonical `NO_GO`.
- Do not stage, commit, merge, overwrite, or delete evidence.

---

### Task 1: Contract and One-Tile Local TDD

**Files:**
- Create: `tools/test_qwen35_real_checkpoint_one_tile_payload_preflight.py`
- Create: `tools/qwen35_real_checkpoint_one_tile_payload_preflight.py`

- [x] Write RED tests for the exact 38-file closure, fixed tile contracts,
  exact payload ranges, two-read hashes, copy isolation, rollback, zero
  execution counters, and TP-specific memory ceilings.
- [x] Run RED and confirm the module is absent.
- [x] Implement tile selection/validation, exact `pread`, BF16 decode,
  independent verifier read, SHA256, copy, isolation, and rollback helpers.
- [x] Use sparse local fixtures to prove short-read/hash/slice/rollback
  failures are fail closed.
- [x] Run focused GREEN and compile.

### Task 2: Fresh-Process Orchestration TDD

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_one_tile_payload_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_one_tile_payload_preflight.py`

- [x] Add RED tests for deterministic 38-file staging, three rank processes,
  fixed no-CUDA/thread environment, partial-failure non-publication, separate
  finalizer, source binding, and atomic two-artifact publication.
- [x] Implement `run`, `internal-rank-worker`, `internal-finalize`, and
  `validate`.
- [x] Run closure, forbidden-call AST, worker rejection, `git diff --check`,
  and staged-file audits.

### Task 3: Live One-Tile Gate and Handoff

**Files:**
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/one_tile_payload_preflight.json`
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/plans/2026-07-28-qwen35-one-convolution-tile-payload-gate.md`

- [x] Run one unique remote gate with three independent worker processes.
- [x] Independently verify source/artifact hashes, ranges, payload byte counts,
  read hashes, destination hashes, isolation, rollback, PIDs, memory, and
  exact remote inventory.
- [x] Run the one-tile, CPU materialization, meta target, construction,
  metadata, worker, factory, binding, loader, authorization, and safety
  regressions.
- [x] Append exact evidence and the next gate boundary to the handoff; mark
  every checkbox only after fresh verification.

## Completion Evidence

Authoritative run:

```text
qwen35-one-tile-payload-20260727-204441
```

Authoritative local artifacts:

```text
experiments/qwen35_hybrid_state/
qwen35-one-tile-payload-20260727-204441/
  one_tile_payload_preflight.json
  source_manifest.json
```

Artifact SHA256:

```text
one_tile_payload_preflight.json:
  1548d743b55ef7276ea5605f69694360ac5ca4272a96bda8d46ba98e18d29c5a
source_manifest.json:
  167e80a001727b5bdc374f313d30ae5359a366c89df99d290542acd60e1af708
```

Fresh rank evidence:

```text
TP=1 rank0:
  PID 2328814
  payload bytes per read 49152
  SHA256 0dbb863f97d7ac62ca2e452e0fe1487edb5d954e2380192102aa1ace8f40642a
  total/post-Torch/post-metadata VmHWM increment:
    4212768 / 3869616 / 3738760 KiB

TP=2 rank0:
  PID 2330411
  payload bytes per read 24576
  SHA256 406a3bd779dbb7a92796e386c2e7843206d399a54e2186edf7d1f7b7f974e1e0
  total/post-Torch/post-metadata VmHWM increment:
    2346752 / 2002972 / 1872048 KiB

TP=2 rank1:
  PID 2331834
  payload bytes per read 24576
  SHA256 a20c9d32c149d39d248c146c65d6ec620b32456423c4d5488b71aeb2cfcc15f4
  total/post-Torch/post-metadata VmHWM increment:
    2346312 / 2002940 / 1871956 KiB
```

Independent verification:

```text
38 local, staged, record, and manifest source hashes matched
source tree SHA256:
  027997c36ef06717c30600cb1b631b6d1d69289f98cee732c8ad21160c1c4f3a
independent remote pread reproduced all three payload hashes
TP2 rank0 bytes + TP2 rank1 bytes == TP1 rank0 bytes
production/verifier/source/destination SHA256 matched per row
non-selected tensors remained zero
rollback restored the selected destination
all unique registered tensors were zero after rollback
loader/assignment/target.take/forward counters remained zero
CUDA remained uninitialized
```

Fresh final verification:

```text
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
frozen 38-file source closure: passed
forbidden-call AST scan: passed
exact two-pread AST scan: passed
real worker direct-execution rejection: passed
git diff --check: passed
staged files: 0
```

Exact claim boundary:

```text
one fixed real layer-0 convolution binding:
  proven for TP=1/2 all ranks
exact rank-local source slicing and squeeze transform:
  proven
independent payload hashes and TP2-to-TP1 concatenation:
  proven
destination isolation and exact rollback:
  proven
any other binding or all-binding loading:
  not executed
checkpoint loader / assignment / target.take():
  not executed
CUDA / Engine / publication / inference:
  absent
production speed / cache / GPU-memory / compression / quality benefit:
  unmeasured
schema-v2 canonical NO_GO:
  unchanged
```
