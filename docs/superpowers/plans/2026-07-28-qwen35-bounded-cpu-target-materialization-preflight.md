# Qwen3.5 Bounded CPU Target Materialization Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize and physically touch the exact TP=1/rank0 and TP=2/rank0-1 empty CPU checkpoint targets in independent processes with zero checkpoint payload, loader, assignment, forward, or CUDA execution.

**Architecture:** Add a source-bound preflight with one rank-worker process per TP row and a separate aggregate finalizer. Each worker builds the exact CPU state pool and CPU model, deduplicates all registered tensor objects, checks exact bytes and binding identity, zero-touches all unique storage, enforces TP-specific memory ceilings, prints one row, and exits.

**Tech Stack:** Python standard library, pathlib, hashlib, json, tarfile, subprocess SSH transport, remote PyTorch, existing bounded metadata and Qwen3.5 target factory.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Use only `sitian@10.232.195.203`.
- Run every TP/rank in a separate fresh Python process.
- TP=1 unique registered bytes must equal `3763656128`.
- TP=2 unique registered bytes must equal `1881936480`.
- TP=1 total/post-Torch/post-metadata ceilings are
  `4718592`/`4194304`/`3932160` KiB.
- TP=2 total/post-Torch/post-metadata ceilings are
  `2621440`/`2359296`/`2097152` KiB.
- Read zero tensor-payload bytes and never recompute the shard SHA256.
- Never call a loader, assignment, `target.take()`, model forward, Engine,
  publication, or restore API.
- Keep CUDA uninitialized.
- Preserve all failed/superseded evidence.
- Do not stage, commit, merge, overwrite, or delete evidence.
- Do not modify schema-v2 canonical `NO_GO`.
- Do not claim runtime performance or quality improvement.

---

### Task 1: Contract and Local Worker TDD

**Files:**
- Create: `tools/test_qwen35_real_checkpoint_cpu_materialization_preflight.py`
- Create: `tools/qwen35_real_checkpoint_cpu_materialization_preflight.py`

- [x] Write RED tests for the exact 32+2 source closure, aggregate/row schemas,
  unique PIDs, exact TP-specific bytes, six rotary buffers, CPU identities,
  zero execution counters, and TP-specific memory ceilings.
- [x] Run the focused test and confirm failure because the module is absent.
- [x] Implement validators, source hashing, namespace loading, memory helpers,
  exact unique-storage accounting, and the local sparse rank worker.
- [x] Verify the worker zero-touches every unique tensor and preserves the
  exact tied embedding, registered destination identities, pool storage, and
  320 binding objects.
- [x] Run focused GREEN and `py_compile`.

### Task 2: Fresh-Process Remote Orchestration TDD

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_cpu_materialization_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_cpu_materialization_preflight.py`

- [x] Add RED tests for deterministic 34-file staging, fixed environment,
  exactly three rank-worker commands, partial-failure non-publication,
  separate finalizer, source-bound round trip, and atomic two-file publish.
- [x] Implement `run`, `internal-rank-worker`, `internal-finalize`, and
  `validate` modes.
- [x] Enforce `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`,
  `CUDA_VISIBLE_DEVICES=""`, `PYTHONDONTWRITEBYTECODE=1`, and `python -B`.
- [x] Run focused GREEN, closure discovery, forbidden-call AST scan,
  direct-worker rejection, `git diff --check`, and staged-file check.

### Task 3: Live Materialization Gate and Handoff

**Files:**
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/cpu_materialization_preflight.json`
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/plans/2026-07-28-qwen35-bounded-cpu-target-materialization-preflight.md`

- [x] Run one unique source-bound remote gate using three independent worker
  processes.
- [x] Independently verify remote inventory, source/artifact hashes, unique
  PIDs, exact bytes, all-zero touch proof, bindings, memory ceilings, and zero
  payload/execution/CUDA.
- [x] Run the new suite plus target-preparation, loader-construction,
  metadata-header, metadata-reader, worker, factory, binding, loader,
  authorization, and safety-gate regressions.
- [x] Run focused compile, closure, forbidden-call, worker-rejection,
  `git diff --check`, and staged-file audits.
- [x] Append exact evidence, failed-run status, claim boundaries, and the next
  safe gate to the handoff; mark checkboxes only after fresh verification.

## Live Result

Authoritative run:

```text
qwen35-cpu-materialization-preflight-20260728-042149
```

Local evidence:

```text
experiments/qwen35_hybrid_state/
  qwen35-cpu-materialization-preflight-20260728-042149/
    cpu_materialization_preflight.json
    source_manifest.json
```

Status:

```text
PASS
```

Fresh-process rows:

```text
TP=1 rank0:
  PID: 2033730
  unique registered/binding bytes:
    3763656128 / 3763655360
  total / post-Torch / post-metadata VmHWM:
    4166176 / 3822432 / 3691344 KiB

TP=2 rank0:
  PID: 2035155
  unique registered/binding bytes:
    1881936480 / 1881935712
  total / post-Torch / post-metadata VmHWM:
    2321784 / 1978028 / 1847896 KiB

TP=2 rank1:
  PID: 2037051
  unique registered/binding bytes:
    1881936480 / 1881935712
  total / post-Torch / post-metadata VmHWM:
    2323192 / 1978356 / 1847952 KiB
```

Every row proved:

```text
metadata bytes read: 144024
payload bytes read: 0
plan loads/skips: 320/312
pool capacity/device/components/bindings/nonzero: 1/cpu/36/0/0
layers/adapters/backends: 24/18/6
bindings/shared/linear/full/buffer/F32: 320/2/252/66/72/36
registered entries/unique tensors/unique binding tensors: 303/302/296
unbound tensors: six exact F32[32] rotary.inv_freq buffers
tied embedding: same Parameter object
all registrations/bindings: CPU
all binding destinations: exact registered objects
all unique tensors zero after touch: true
loader/assignment/model-forward/attention-forward calls: 0/0/0/0
CUDA initialized before/after: false/false
```

Remote inventory was exactly 34 staged source files plus:

```text
cpu_materialization_preflight.json
source_manifest.json
```

All local, remote, record, and manifest source hashes matched.

Artifact SHA256 values:

```text
cpu_materialization_preflight.json:
  34f533d37e25afa534f9219db019edaabbdf9fcbaab8208fa5d81934aac51611
source_manifest.json:
  48dcbd9603b4c3f49809fd6fdcfccbb43b4f763186e662cf4cf2507da84e3d9e
```

Fresh final verification:

```text
CPU materialization preflight: 6 tests
meta target preparation preflight: 7 tests
loader construction preflight: 5 tests
metadata-header preflight: 6 tests
bounded metadata reader: 4 tests
worker helper: 6 tests
candidate factory: 6 tests
real component binding: 1 test
loader configuration: 4 tests
candidate loader: 5 tests
authorization: passed
safety gate: 23 tests
focused py_compile: passed
frozen 32+2 source closure: passed
forbidden-call AST scan: passed
real worker direct-execution rejection: passed
git diff --check: passed
staged files: 0
```

Exact claim boundary:

```text
approved metadata to physically committed empty CPU target:
  proven for TP=1/2 all ranks
exact rank-local bytes, tied embedding, pool and 320 binding identity:
  proven
fresh-process bounded host memory:
  proven
checkpoint payload loading / loader call / assignment:
  not executed
target.take() / candidate load / Engine installation:
  not executed
CUDA / publication / inference:
  absent
production speed / cache / GPU-memory / compression / quality benefit:
  unmeasured
schema-v2 canonical NO_GO:
  unchanged
```
