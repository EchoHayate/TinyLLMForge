# Qwen3.5 Meta Target-Preparation Memory Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare the exact TP=1/rank0 and TP=2/rank0-1 Qwen3.5 checkpoint targets in independent remote processes using only a capacity-one CPU hybrid-state pool and `meta` model parameters, with zero payload, loader, assignment, forward, or CUDA execution.

**Architecture:** Add one standalone source-bound preflight module with a rank-worker mode and an aggregate finalizer. The orchestrator stages the frozen production closure, launches one process per rank, validates exact pool/graph/binding contracts and three layered VmHWM ceilings, then publishes two JSON artifacts only after all rows pass.

**Tech Stack:** Python standard library, pathlib, hashlib, json, tarfile, subprocess SSH transport, remote PyTorch, existing Qwen3.5 bounded metadata, hybrid-state, component-factory, and checkpoint-binding modules.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Use only `sitian@10.232.195.203` through `/tmp/ssh-sitian-10.232.195.203`.
- Each TP/rank row must run in an independent fresh remote Python process.
- Enforce total/post-Torch/post-metadata VmHWM ceilings of
  `524288`/`196608`/`32768` KiB.
- Read zero tensor-payload bytes and never recompute the full shard SHA256.
- Allocate only the exact capacity-one CPU state pool; all model parameters and checkpoint buffers must be `meta`.
- Require TP=1 logical/physical pool bytes `10321920` and TP=2
  logical/physical pool bytes `5160960`.
- Never call a loader, assignment function, `target.take()`, model forward, Engine, publication, or restore API.
- Keep CUDA uninitialized and set `CUDA_VISIBLE_DEVICES=""`.
- Do not invoke or enable the real checkpoint-load worker `main()`.
- Preserve failed and superseded local/remote evidence.
- Do not stage, commit, merge, overwrite, or delete evidence.
- Do not modify or reinterpret schema-v2 canonical `NO_GO`.
- Do not claim performance, cache, GPU-memory, compression, or quality improvement.

---

### Task 1: Freeze the Record and Fresh-Process Contract

**Files:**
- Create: `tools/test_qwen35_real_checkpoint_target_preparation_preflight.py`
- Create: `tools/qwen35_real_checkpoint_target_preparation_preflight.py`

**Interfaces:**
- Consumes: existing `read_qwen35_checkpoint_metadata`,
  `build_qwen35_checkpoint_tensor_plan`,
  `build_qwen35_hybrid_state_layout`,
  `HybridStateTensorPool`, and
  `prepare_qwen35_checkpoint_candidate_target`.
- Produces: `validate_target_preparation_row(record)`,
  `validate_target_preparation_preflight(record)`,
  `run_target_preparation_rank_worker(...)`, and fixed schema/source constants.

- [x] Write RED tests that import the absent module and freeze the exact
  production closure, three TP rows, aggregate schema, identity fields,
  unique process IDs, zero execution counters, expected pool bytes, exact
  graph/binding counts, CUDA state, and all three memory ceilings.
- [x] Run
  `/opt/homebrew/bin/python3.12 tools/test_qwen35_real_checkpoint_target_preparation_preflight.py`
  and confirm failure because the module or required interfaces are absent.
- [x] Implement constants, deterministic source hashing/tree identity,
  namespace-package installation, row validation, aggregate validation, and
  memory-delta helpers without adding any remote execution.
- [x] Re-run the focused test and confirm the contract tests pass while worker
  behavior tests remain RED.

### Task 2: Implement the Local Sparse Rank Worker Through TDD

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_target_preparation_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_target_preparation_preflight.py`

**Interfaces:**
- Consumes: Task 1 validators and a sparse local shard containing the retained
  real header.
- Produces: one validated rank-row mapping per fresh worker invocation.

- [x] Add RED tests that build the real 24-layer graph from retained metadata
  snapshots and assert exact pool bytes, unchanged pool tensors, backend
  calls, 320 meta bindings, registered tensor devices, release sampling,
  zero payload/loader/assignment/forward events, and CUDA false.
- [x] Run the focused test and confirm failure at the missing rank-worker
  behavior.
- [x] Implement bounded metadata reading, exact plan/pool construction,
  non-executing attention backends, post-metadata safetensors-open guard,
  graph/binding/pool inspection, immediate release, `gc.collect()`, and six
  memory samples.
- [x] Re-run the focused test and confirm all local worker tests pass.

### Task 3: Implement Deterministic Remote Orchestration Through TDD

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_target_preparation_preflight.py`
- Modify: `tools/qwen35_real_checkpoint_target_preparation_preflight.py`

**Interfaces:**
- Consumes: one unique run tag and the Task 2 rank-worker CLI.
- Produces: `execute_remote_target_preparation_preflight(...)` and exactly two
  source-bound local artifacts after all three rows pass.

- [x] Add RED tests for deterministic 33-file tar staging, fixed SSH command,
  no-bytecode/CUDA environment, one command per TP row, no local publication
  on a partial failure, remote finalization, artifact round trip, and atomic
  two-file local publication.
- [x] Run the focused test and confirm orchestration assertions fail.
- [x] Implement unique remote staging, source SHA256 verification, sequential
  fresh-process rank execution, row aggregation, separate remote finalizer,
  round-trip verification, and atomic local publication.
- [x] Add CLI modes `run`, `internal-rank-worker`, `internal-finalize`, and
  `validate`; reject unapproved checkpoint paths before production imports or
  reads.
- [x] Re-run the focused test, focused `py_compile`, closure discovery, static
  forbidden-call scan, and `git diff --check`.

### Task 4: Run and Independently Verify the Live Gate

**Files:**
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/target_preparation_preflight.json`
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/plans/2026-07-28-qwen35-meta-target-preparation-memory-preflight.md`

**Interfaces:**
- Consumes: the approved model directory and the complete preflight module.
- Produces: one authoritative source-bound PASS or a preserved failed remote
  run with no local authoritative publication.

- [x] Rebuild/check the `sitian` ControlMaster with
  `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- [x] Run one unique remote preflight tag; do not reuse the non-authoritative
  probe directory.
- [x] Independently verify exact remote inventory, source hashes, artifact
  hashes, unique PIDs, TP rows, pool bytes, graph/binding counts, zero
  execution/payload/CUDA, and all memory ceilings.
- [x] Run the focused target-preparation test plus bounded metadata, candidate
  factory, real component binding, loader construction, loader configuration,
  candidate loader, worker helper, authorization, and safety-gate regressions.
- [x] Run focused `py_compile`, frozen closure check, forbidden-call scans,
  direct real-worker rejection, `git diff --check`, and staged-file check.
- [x] Append exact evidence, failed/superseded run status, claim boundaries,
  and the next safe gate to `AGENT_HANDOFF_STATE.md`.
- [x] Mark every completed checkbox only after fresh verification evidence is
  available.

## Live Result

Authoritative run:

```text
qwen35-target-preparation-preflight-final-20260728-040143
```

Local evidence:

```text
experiments/qwen35_hybrid_state/
  qwen35-target-preparation-preflight-final-20260728-040143/
    target_preparation_preflight.json
    source_manifest.json
```

Status:

```text
PASS
```

Fresh-process rows:

```text
TP=1 rank0:
  PID: 1775726
  pool bytes: 10321920
  total / post-Torch / post-metadata VmHWM increment:
  491436 / 145448 / 14728 KiB

TP=2 rank0:
  PID: 1776908
  pool bytes: 5160960
  total / post-Torch / post-metadata VmHWM increment:
  485828 / 139852 / 9800 KiB

TP=2 rank1:
  PID: 1778351
  pool bytes: 5160960
  total / post-Torch / post-metadata VmHWM increment:
  486328 / 140720 / 10472 KiB
```

Every row proved:

```text
metadata bytes read: 144024
payload bytes read: 0
plan loads/skips: 320/312
layers/adapters/backends: 24/18/6
bindings/shared/linear/full/buffer/F32:
  320/2/252/66/72/36
unexpected non-meta registrations: []
pool bindings/nonzero values: 0/0
loader/assignment/model-forward/attention-forward calls: 0/0/0/0
CUDA initialized before/after: false/false
```

The first formal run:

```text
qwen35-target-preparation-preflight-20260728-035940
```

failed closed and published no local artifact. Rank JSON was serialized with
sorted keys, but the validator incorrectly required insertion order for the
six `memory` fields. A JSON round-trip regression test reproduced the failure.
The validator now requires the exact key set and still reads every named stage
explicitly. The failed remote directory contains only the 33 staged sources.

The authoritative remote inventory is exactly 33 staged source files plus:

```text
target_preparation_preflight.json
source_manifest.json
```

All local source hashes, remote source hashes, record hashes, and source
manifest hashes match.

Artifact SHA256 values:

```text
target_preparation_preflight.json:
  8752002c79d18488bc338bc597a1d3e55e90c758d975ad826837248f9dfdd352
source_manifest.json:
  efef55ad6cdfeeb7ab884eb8c6e72163ae1ec3d1c6e5eb75d2674ba4b625487a
```

Fresh final verification:

```text
target preparation preflight: 7 tests
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
frozen 32+1 source closure: passed
forbidden-call AST scan: passed
real worker direct-execution rejection: passed
git diff --check: passed
staged files: 0
```

Exact claim boundary:

```text
approved metadata to exact CPU pool + real meta graph + 320 bindings:
  proven remotely for TP=1/2 all ranks
fresh-process and bounded host-memory preparation:
  proven
payload loading, loader call, assignment, target consumption:
  not executed
CPU/GPU model materialization:
  not executed
CUDA/Engine/publication/inference:
  absent
production speed/cache/GPU-memory/compression/quality benefit:
  unmeasured
schema-v2 canonical NO_GO:
  unchanged
```
